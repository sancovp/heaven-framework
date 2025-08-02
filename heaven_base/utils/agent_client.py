"""
AgentClient Library

Generic client wrapper for any BaseHeavenAgent-based agent.
Allows synchronous, asynchronous, batch, retry, resume, and scheduling.
"""
import asyncio
import logging
from typing import Any, Callable, List, Dict, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from dateutil.parser import parse
from .memory.history import History
from .unified_chat import UnifiedChat
from .baseheavenagent import BaseHeavenAgent, HeavenAgentConfig

# Setup default logger
logger = logging.getLogger('AgentClient')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class AgentClient:
    """
    A generic client wrapper for any Agent based on BaseHeavenAgent.

    Constructor takes either:
      - agent_factory: a callable that returns a new BaseHeavenAgent instance
      - or a (config, unified_chat, history) tuple to create a new agent per call
    """

    def __init__(
        self,
        agent_factory: Optional[Callable[[], BaseHeavenAgent]] = None,
        config: Optional[HeavenAgentConfig] = None,
        history: Optional[History] = None,
        iterations: int = 1
    ):
        if not agent_factory and not config:
            raise ValueError("Either agent_factory or config must be provided")
        self.iterations = iterations
        self.history = history or History(messages=[])
        self.unified_chat = UnifiedChat()
        if agent_factory:
            self.agent_factory = agent_factory
        else:
            # build factory from config
            def _factory():
                return BaseHeavenAgent(
                    config=config,
                    unified_chat=self.unified_chat,
                    history=self.history
                )
            self.agent_factory = _factory
        # background scheduler
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        self.jobs: Dict[str, Any] = {}

    async def _async_call(self, prompt: str, iterations: Optional[int] = None) -> Dict:
        """
        Internal async call: spins up a fresh agent and runs one or more iterations.
        """
        iters = iterations or self.iterations
        agent = self.agent_factory()
        prompt_text = prompt if prompt.strip().startswith('agent goal=') else f"agent goal={prompt}, iterations={iters}"
        return await agent.run(prompt=prompt_text)

    def call(self, prompt: str, iterations: Optional[int] = None) -> str:
        """
        Synchronous call: returns last AIMessage content or raw result.
        """
        response = asyncio.run(self._async_call(prompt, iterations))
        history = response.get('history')
        if history and hasattr(history, 'messages'):
            # find last AIMessage
            for msg in reversed(history.messages):
                if msg.__class__.__name__ == 'AIMessage':
                    return msg.content
        return str(response)

    def batch(self, prompts: List[str], iterations: Optional[int] = None) -> List[str]:
        """
        Run multiple prompts sequentially.
        """
        return [self.call(p, iterations) for p in prompts]

    def retry(
        self,
        prompt: str,
        iterations: Optional[int] = None,
        retries: int = 3,
        backoff_factor: float = 2.0
    ) -> str:
        """
        Retryable call with exponential backoff.
        """
        delay = 1.0
        for attempt in range(1, retries + 1):
            try:
                return self.call(prompt, iterations)
            except Exception as e:
                if attempt == retries:
                    logger.error(f"All {retries} attempts failed: {e}")
                    raise
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay}s...")
                import time; time.sleep(delay)
                delay *= backoff_factor

    def resume(self, prompt: str, history_id: str) -> Dict:
        """
        Continue a prior conversation by history_id.
        """
        agent = self.agent_factory()
        return asyncio.run(agent.run(prompt=prompt, history_id=history_id))

    def schedule(
        self,
        task_name: str,
        prompt: str,
        run_at: Optional[str] = None,
        cron: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Schedule a one-time or cron-based job.
        """
        if cron:
            trigger = CronTrigger.from_crontab(cron)
            job = self.scheduler.add_job(
                self.call,
                trigger=trigger,
                args=[prompt],
                id=task_name
            )
        else:
            run_date = parse(run_at) if run_at else None
            job = self.scheduler.add_job(
                self.call,
                'date',
                run_date=run_date,
                args=[prompt],
                id=task_name
            )
        self.jobs[task_name] = job
        return {"task": task_name, "next_run": str(job.next_run_time)}

    def list_jobs(self) -> List[str]:
        return list(self.jobs.keys())
