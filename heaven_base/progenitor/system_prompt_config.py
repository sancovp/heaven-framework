import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, Type, List
import importlib
import sys
import logging
from ..utils.get_env_value import EnvConfigUtil

# Set up logging
logger = logging.getLogger(__name__)

class SystemPromptConfig:
    """
    Configuration for building system prompts dynamically.
    Looks up the latest version of agent_dna and builds prompts on demand.
    """

    def __init__(self, species_name: str, agent_type: str, agent_name: str, 
                 domain: str, process: str, 
                 base_dir: str = None
                ):
        """
        Initialize a system prompt configuration.

        Args:
            species_name: Name of the species (e.g., "HeavenlyBeing")
            agent_type: Type of agent ("deity", "progenitor", or "worker")
            agent_name: Name of the specific agent
            domain: Domain the agent operates in
            process: Process the agent is responsible for
            base_dir: Base directory for the prompt engineering system
        """
        self.species_name = species_name
        self.agent_type = agent_type.lower()  # normalize to lowercase
        self.agent_name = agent_name
        self.domain = domain.strip() if domain else ""
        self.process = process.strip() if process else ""
        if base_dir is None:
            base_dir = os.path.join(EnvConfigUtil.get_heaven_data_dir(), "progenitor")
        self.base_dir = base_dir

        # Validate agent type
        if self.agent_type not in ["deity", "progenitor", "worker"]:
            raise ValueError(f"Invalid agent_type: {self.agent_type}. Must be 'deity', 'progenitor', or 'worker'")

    def get_safe_names(self):
        """Get path-safe versions of names."""
        domain_safe = self.domain.replace(" ", "_").lower() if self.domain else ""
        process_safe = self.process.replace(" ", "_").lower() if self.process else ""
        return domain_safe, process_safe

    def get_agent_dna_path(self) -> str:
        """Get the path to the most recent agent DNA file."""
        domain_safe, process_safe = self.get_safe_names()

        # Determine base DNA filename by agent type
        if self.agent_type == "deity":
            base_dna_filename = f"{self.species_name}_deity_dna.json"
            base_dna_dir = ""  # No subdirectory
        elif self.agent_type == "progenitor":
            if not self.domain or self.domain == "default":
                # Default progenitor
                base_dna_filename = f"{self.species_name}_default_progenitor_dna.json"
                base_dna_dir = ""  # No subdirectory
            else:
                # Domain-specific progenitor
                base_dna_filename = f"{self.species_name}_{domain_safe}_progenitor_dna.json"
                base_dna_dir = domain_safe  # Domain subdirectory
        else:  # worker
            base_dna_filename = f"{self.species_name}_{domain_safe}_{process_safe}_dna.json"
            base_dna_dir = domain_safe  # Domain subdirectory

        # Construct paths to check for evolved or original DNA
        if base_dna_dir:
            # With domain subdirectory
            evolved_path = os.path.join(
                self.base_dir, "species", self.species_name, "evolved_agent_dna", 
                base_dna_dir, base_dna_filename
            )
            original_path = os.path.join(
                self.base_dir, "species", self.species_name, "agent_dna", 
                base_dna_dir, base_dna_filename
            )
        else:
            # No domain subdirectory (deity or default progenitor)
            evolved_path = os.path.join(
                self.base_dir, "species", self.species_name, "evolved_agent_dna",
                base_dna_filename
            )
            original_path = os.path.join(
                self.base_dir, "species", self.species_name, "agent_dna",
                base_dna_filename
            )

        # Use evolved if it exists, otherwise use original
        if os.path.exists(evolved_path):
            return evolved_path
        else:
            return original_path

    def build(self):
        """
        Build the system prompt from scratch each time.

        Returns:
            The complete system prompt string
        """
        # Get the agent DNA path
        agent_dna_path = self.get_agent_dna_path()

        # Load the agent DNA
        try:
            with open(agent_dna_path, 'r') as f:
                agent_dna = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Agent DNA file not found at {agent_dna_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in agent DNA file at {agent_dna_path}")

        # Ensure species path is in sys.path
        species_path = os.path.join(self.base_dir, "species", self.species_name)
        if species_path not in sys.path:
            sys.path.append(os.path.dirname(species_path))

        # Dynamically import all required modules for the species
        module_prefix = f"{self.species_name}."

        # Build class names based on naming convention
        world_class_name = f"{self.species_name}WorldSettings"
        egregore_class_name = f"{self.species_name}EgregoreSettings"
        deity_class_name = f"{self.species_name}DeitySettings"
        progenitor_class_name = f"{self.species_name}ProgenitorSettings1"
        agent_class_name = f"{self.species_name}AgentSettings"

        # Import modules with importlib to ensure we get the latest version
        world_module = importlib.import_module(
            f"{module_prefix}{world_class_name.lower()}"
        )
        egregore_module = importlib.import_module(
            f"{module_prefix}{egregore_class_name.lower()}"
        )
        deity_module = importlib.import_module(
            f"{module_prefix}{deity_class_name.lower()}"
        )
        progenitor_module = importlib.import_module(
            f"{module_prefix}{progenitor_class_name.lower()}"
        )
        agent_module = importlib.import_module(
            f"{module_prefix}{agent_class_name.lower()}"
        )

        # Get the actual class objects
        WorldSettings = getattr(world_module, world_class_name)
        EgregoreSettings = getattr(egregore_module, egregore_class_name)
        DeitySettings = getattr(deity_module, deity_class_name)
        ProgenitorSettings = getattr(progenitor_module, progenitor_class_name)
        AgentSettings = getattr(agent_module, agent_class_name)

        # Create settings instances
        world_settings = WorldSettings()
        egregore_settings = EgregoreSettings(world_settings=world_settings)
        deity_settings = DeitySettings(
            world_settings=world_settings, 
            egregore_settings=egregore_settings
        )
        progenitor_settings = ProgenitorSettings(
            world_settings=world_settings, 
            egregore_settings=egregore_settings,
            deity_settings=deity_settings
        )

        # Import Profile and ProfileMaker dynamically
        # meta_settings = importlib.import_module("heaven_base.progenitor.meta_settings") # Not implemented yet
        ProfileMaker = getattr(meta_settings, "ProfileMaker")
        if ProfileMaker:
            profile_maker = ProfileMaker(
                world_settings=world_settings,
                egregore_settings=egregore_settings,
                deity_settings=deity_settings,
                progenitor_settings=progenitor_settings,
                agent_settings_class=AgentSettings
            )

            if self.agent_type == "deity":
                prompt = profile_maker.create_deity_profile(agent_dna)
            elif self.agent_type == "progenitor":
                prompt = profile_maker.create_progenitor_profile(agent_dna)
            else:  # worker
                prompt = profile_maker.create_worker_profile(agent_dna)
            if prompt:
                return prompt
        else:
            return f"{self.agent_name}: SystemPromptConfig failed to use ProfileMaker"

    def revert_evolutions(self) -> bool:
        """
        Revert to original agent DNA by moving evolved DNA to quarantine.

        Returns:
            True if reversion occurred, False if no evolved DNA exists
        """
        domain_safe, process_safe = self.get_safe_names()

        # Determine base DNA filename by agent type
        if self.agent_type == "deity":
            base_dna_filename = f"{self.species_name}_deity_dna.json"
            base_dna_dir = ""  # No subdirectory
        elif self.agent_type == "progenitor":
            if not self.domain or self.domain == "default":
                # Default progenitor
                base_dna_filename = f"{self.species_name}_default_progenitor_dna.json"
                base_dna_dir = ""  # No subdirectory
            else:
                # Domain-specific progenitor
                base_dna_filename = f"{self.species_name}_{domain_safe}_progenitor_dna.json"
                base_dna_dir = domain_safe  # Domain subdirectory
        else:  # worker
            base_dna_filename = f"{self.species_name}_{domain_safe}_{process_safe}_dna.json"
            base_dna_dir = domain_safe  # Domain subdirectory

        # Construct path to evolved DNA
        if base_dna_dir:
            # With domain subdirectory
            evolved_path = os.path.join(
                self.base_dir, "species", self.species_name, "evolved_agent_dna", 
                base_dna_dir, base_dna_filename
            )
            quarantine_dir = os.path.join(
                self.base_dir, "species", self.species_name, "quarantined_agent_dna",
                base_dna_dir
            )
        else:
            # No domain subdirectory (deity or default progenitor)
            evolved_path = os.path.join(
                self.base_dir, "species", self.species_name, "evolved_agent_dna",
                base_dna_filename
            )
            quarantine_dir = os.path.join(
                self.base_dir, "species", self.species_name, "quarantined_agent_dna"
            )

        # Check if evolved DNA exists
        if os.path.exists(evolved_path):
            # Create quarantine directory if it doesn't exist
            os.makedirs(quarantine_dir, exist_ok=True)

            # Generate timestamped quarantine filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_file = os.path.join(
                quarantine_dir, 
                f"{timestamp}_{os.path.basename(evolved_path)}"
            )

            # Move the evolved DNA to quarantine
            shutil.move(evolved_path, quarantine_file)

            # Log the reversion
            logger.warning(
                f"Agent {self.agent_name} DNA reverted to original. "
                f"Evolved DNA quarantined to {quarantine_file}"
            )

            return True

        return False

    def save_evolution(self, new_agent_dna: Dict[str, Any]) -> str:
        """
        Save evolved agent DNA.

        Args:
            new_agent_dna: The new agent DNA to save

        Returns:
            Path to the saved evolved DNA file
        """
        domain_safe, process_safe = self.get_safe_names()

        # Determine base DNA filename and directory by agent type
        if self.agent_type == "deity":
            base_dna_filename = f"{self.species_name}_deity_dna.json"
            base_dna_dir = ""  # No subdirectory
        elif self.agent_type == "progenitor":
            if not self.domain or self.domain == "default":
                # Default progenitor
                base_dna_filename = f"{self.species_name}_default_progenitor_dna.json"
                base_dna_dir = ""  # No subdirectory
            else:
                # Domain-specific progenitor
                base_dna_filename = f"{self.species_name}_{domain_safe}_progenitor_dna.json"
                base_dna_dir = domain_safe  # Domain subdirectory
        else:  # worker
            base_dna_filename = f"{self.species_name}_{domain_safe}_{process_safe}_dna.json"
            base_dna_dir = domain_safe  # Domain subdirectory

        # Construct path to evolved DNA
        if base_dna_dir:
            # With domain subdirectory
            evolved_dir = os.path.join(
                self.base_dir, "species", self.species_name, "evolved_agent_dna", 
                base_dna_dir
            )
            evolved_path = os.path.join(evolved_dir, base_dna_filename)
        else:
            # No domain subdirectory (deity or default progenitor)
            evolved_dir = os.path.join(
                self.base_dir, "species", self.species_name, "evolved_agent_dna"
            )
            evolved_path = os.path.join(evolved_dir, base_dna_filename)

        # Create directory if it doesn't exist
        os.makedirs(evolved_dir, exist_ok=True)

        # Save the evolved DNA
        with open(evolved_path, 'w') as f:
            json.dump(new_agent_dna, f, indent=2)

        logger.info(f"Saved evolved DNA for {self.agent_name} to {evolved_path}")
        return evolved_path

# import os
# import json
# import shutil
# from datetime import datetime
# from typing import Dict, Any, Optional, Type, List
# import importlib
# import sys
# import logging

# # Set up logging
# logger = logging.getLogger(__name__)

# class SystemPromptConfig:
#     """
#     Configuration for building system prompts dynamically.
#     Looks up the latest version of agent_dna and builds prompts on demand.
#     """

#     def __init__(self, species_name: str, agent_type: str, agent_name: str, 
#                  domain: str, process: str, base_dir: str = os.path.join(os.path.expanduser("~"), ".heaven", "prompt_engineering_system")):
#         """
#         Initialize a system prompt configuration.

#         Args:
#             species_name: Name of the species (e.g., "HeavenlyBeing")
#             agent_type: Type of agent ("deity", "progenitor", or "worker")
#             agent_name: Name of the specific agent
#             domain: Domain the agent operates in
#             process: Process the agent is responsible for
#             base_dir: Base directory for the prompt engineering system
#         """
#         self.species_name = species_name
#         self.agent_type = agent_type
#         self.agent_name = agent_name
#         self.domain = domain
#         self.process = process
#         self.base_dir = base_dir

#     def get_agent_dna_path(self) -> str:
#         """Get the path to the most recent agent DNA file."""
#         # Format directory paths
#         domain_safe = self.domain.replace(" ", "_").lower()
#         process_safe = self.process.replace(" ", "_").lower()
#         agent_safe = self.agent_name.replace(" ", "_").lower()

#         # For deity and progenitor, don't include process in path
#         if self.agent_type in ["deity", "progenitor"]:
#             evolved_path = os.path.join(
#                 self.base_dir, "species", self.species_name, "evolved_agent_dna",
#                 f"{self.species_name}_{self.agent_type}_{agent_safe}_dna.json"
#             )
#             original_path = os.path.join(
#                 self.base_dir, "species", self.species_name, "agent_dna",
#                 f"{self.species_name}_{self.agent_type}_{agent_safe}_dna.json"
#             )
#         else:
#             # Worker agents include domain and process
#             evolved_path = os.path.join(
#                 self.base_dir, "species", self.species_name, "evolved_agent_dna", domain_safe,
#                 f"{self.species_name}_{domain_safe}_{process_safe}_{agent_safe}_dna.json"
#             )
#             original_path = os.path.join(
#                 self.base_dir, "species", self.species_name, "agent_dna", domain_safe,
#                 f"{self.species_name}_{domain_safe}_{process_safe}_{agent_safe}_dna.json"
#             )

#         # Use evolved if it exists, otherwise use original
#         if os.path.exists(evolved_path):
#             return evolved_path
#         else:
#             return original_path

#     def build(self):
#         """
#         Build the system prompt from scratch each time.

#         Returns:
#             The complete system prompt string
#         """
#         # Get the agent DNA path
#         agent_dna_path = self.get_agent_dna_path()

#         # Load the agent DNA
#         try:
#             with open(agent_dna_path, 'r') as f:
#                 agent_dna = json.load(f)
#         except FileNotFoundError:
#             raise ValueError(f"Agent DNA file not found at {agent_dna_path}")
#         except json.JSONDecodeError:
#             raise ValueError(f"Invalid JSON in agent DNA file at {agent_dna_path}")

#         # Ensure species path is in sys.path
#         species_path = os.path.join(self.base_dir, "species", self.species_name)
#         if species_path not in sys.path:
#             sys.path.append(os.path.dirname(species_path))

#         # Dynamically import all required modules for the species
#         module_prefix = f"{self.species_name}."

#         # Build class names based on naming convention
#         world_class_name = f"{self.species_name}WorldSettings"
#         egregore_class_name = f"{self.species_name}EgregoreSettings"
#         deity_class_name = f"{self.species_name}DeitySettings"
#         progenitor_class_name = f"{self.species_name}ProgenitorSettings"
#         agent_class_name = f"{self.species_name}AgentSettings"

#         # Import modules with importlib to ensure we get the latest version
#         world_module = importlib.import_module(
#             f"{module_prefix}{world_class_name.lower()}"
#         )
#         egregore_module = importlib.import_module(
#             f"{module_prefix}{egregore_class_name.lower()}"
#         )
#         deity_module = importlib.import_module(
#             f"{module_prefix}{deity_class_name.lower()}"
#         )
#         progenitor_module = importlib.import_module(
#             f"{module_prefix}{progenitor_class_name.lower()}"
#         )
#         agent_module = importlib.import_module(
#             f"{module_prefix}{agent_class_name.lower()}"
#         )

#         # Get the actual class objects
#         WorldSettings = getattr(world_module, world_class_name)
#         EgregoreSettings = getattr(egregore_module, egregore_class_name)
#         DeitySettings = getattr(deity_module, deity_class_name)
#         ProgenitorSettings = getattr(progenitor_module, progenitor_class_name)
#         AgentSettings = getattr(agent_module, agent_class_name)

#         # Create settings instances
#         world_settings = WorldSettings()
#         egregore_settings = EgregoreSettings(world_settings=world_settings)
#         deity_settings = DeitySettings(
#             world_settings=world_settings, 
#             egregore_settings=egregore_settings
#         )
#         progenitor_settings = ProgenitorSettings(
#             world_settings=world_settings, 
#             egregore_settings=egregore_settings,
#             deity_settings=deity_settings
#         )

#         # Import Profile and ProfileMaker dynamically
#         meta_settings = importlib.import_module("meta_settings")
#         ProfileMaker = getattr(meta_settings, "ProfileMaker")
#         if ProfileMaker:
#             profile_maker = ProfileMaker(
#                 world_settings=world_settings,
#                 egregore_settings=egregore_settings,
#                 deity_settings=deity_settings,
#                 progenitor_settings=progenitor_settings,
#                 agent_settings_class=AgentSettings
#             )
    
#             if self.agent_type == "deity":
#                 prompt = profile_maker.create_deity_profile(agent_dna)
#             elif self.agent_type == "progenitor":
#                 prompt = profile_maker.create_progenitor_profile(agent_dna)
#             else:  # worker
#                 prompt = profile_maker.create_worker_profile(agent_dna)
#             if prompt:
#                 return prompt
#         else:
#             return f"{self.agent_name}: SystemPromptConfig failed to use ProfileMaker"

#     def revert_evolutions(self) -> bool:
#         """
#         Revert to original agent DNA by moving evolved DNA to quarantine.

#         Returns:
#             True if reversion occurred, False if no evolved DNA exists
#         """
#         # Determine the evolved path based on agent type
#         domain_safe = self.domain.replace(" ", "_").lower()
#         process_safe = self.process.replace(" ", "_").lower()
#         agent_safe = self.agent_name.replace(" ", "_").lower()

#         if self.agent_type in ["deity", "progenitor"]:
#             evolved_path = os.path.join(
#                 self.base_dir, "species", self.species_name, "evolved_agent_dna",
#                 f"{self.species_name}_{self.agent_type}_{agent_safe}_dna.json"
#             )
#             quarantine_dir = os.path.join(
#                 self.base_dir, "species", self.species_name, "quarantined_agent_dna"
#             )
#         else:
#             evolved_path = os.path.join(
#                 self.base_dir, "species", self.species_name, "evolved_agent_dna", domain_safe,
#                 f"{self.species_name}_{domain_safe}_{process_safe}_{agent_safe}_dna.json"
#             )
#             quarantine_dir = os.path.join(
#                 self.base_dir, "species", self.species_name, "quarantined_agent_dna", domain_safe
#             )

#         # Check if evolved DNA exists
#         if os.path.exists(evolved_path):
#             # Create quarantine directory if it doesn't exist
#             os.makedirs(quarantine_dir, exist_ok=True)

#             # Generate timestamped quarantine filename
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             quarantine_file = os.path.join(
#                 quarantine_dir, 
#                 f"{timestamp}_{os.path.basename(evolved_path)}"
#             )

#             # Move the evolved DNA to quarantine
#             shutil.move(evolved_path, quarantine_file)

#             # Log the reversion
#             logger.warning(
#                 f"Agent {self.agent_name} DNA reverted to original. "
#                 f"Evolved DNA quarantined to {quarantine_file}"
#             )

#             return True

#         return False

#     def save_evolution(self, new_agent_dna: Dict[str, Any]) -> str:
#         """
#         Save evolved agent DNA.

#         Args:
#             new_agent_dna: The new agent DNA to save

#         Returns:
#             Path to the saved evolved DNA file
#         """
#         # Format directory paths
#         domain_safe = self.domain.replace(" ", "_").lower()
#         process_safe = self.process.replace(" ", "_").lower()
#         agent_safe = self.agent_name.replace(" ", "_").lower()

#         # Determine the path based on agent type
#         if self.agent_type in ["deity", "progenitor"]:
#             evolved_dir = os.path.join(
#                 self.base_dir, "species", self.species_name, "evolved_agent_dna"
#             )
#             evolved_path = os.path.join(
#                 evolved_dir,
#                 f"{self.species_name}_{self.agent_type}_{agent_safe}_dna.json"
#             )
#         else:
#             # Worker agents include domain and process
#             evolved_dir = os.path.join(
#                 self.base_dir, "species", self.species_name, "evolved_agent_dna", domain_safe
#             )
#             evolved_path = os.path.join(
#                 evolved_dir,
#                 f"{self.species_name}_{domain_safe}_{process_safe}_{agent_safe}_dna.json"
#             )

#         # Create directory if it doesn't exist
#         os.makedirs(evolved_dir, exist_ok=True)

#         # Save the evolved DNA
#         with open(evolved_path, 'w') as f:
#             json.dump(new_agent_dna, f, indent=2)

#         logger.info(f"Saved evolved DNA for {self.agent_name} to {evolved_path}")
#         return evolved_path