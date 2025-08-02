from typing import Dict, Any, Optional
from ..baseheaventool import BaseHeavenTool, ToolArgsSchema
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from ..tool_utils._think_utils import think_process

class ThinkToolArgsSchema(ToolArgsSchema):
    arguments: Dict[str, Dict[str, Any]] = {
        'thoughts': {
            'name': 'thoughts',            'type': 'str',            'description': 'The detailed thoughts or reasoning process about something',            'required': True        },        'conclusion': {
            'name': 'conclusion',            'type': 'str',            'description': 'The final conclusion or insight derived from the thoughts above',            'required': False        }    }

class ThinkTool(BaseHeavenTool):
    name = "ThinkTool"
    description = """A tool for stopping and thinking before continuing (taking a thinking turn). The thoughts do not need to be repeated. Thinking turns can be chained together before giving a final response -- this is called a Chain of Thought.

## Info
The ThinkTool can be used in specific sequences before responding to the user as a way to create emergent cognitive architectures, like building thought protocols or mental algorithms: 

#### CoT format
🧠⚙️ means 'ThinkTool' use turn. CoTs represent multiple ThinkTool uses before responding to user
"LikeX" means 'in the memeplex of'

#### CoT Representation Syntax and Interaction Flow
```
// user input ->
CoT(Name:[
  🧠⚙️(type[phases]) ->  
  <...rest of sequence...>
]) ->
// final response
```

## Examples
#### Common frameworks
CoT(Dialectic:[
  🧠⚙️(type: Thesis) -> 
  🧠⚙️(type: Antithesis) -> 
  🧠⚙️(type: Synthesis)
])

CoT(SixHats:[
  🧠⚙️(type: WhiteHat_Facts) -> 
  🧠⚙️(type: RedHat_Emotions) -> 
  🧠⚙️(type: BlackHat_Caution) -> 
  🧠⚙️(type: YellowHat_Benefits) -> 
  🧠⚙️(type: GreenHat_Creativity) -> 
  🧠⚙️(type: BlueHat_Process)
])

CoT(DesignThinking:[
  🧠⚙️(type: Empathize) -> 
  🧠⚙️(type: Define) -> 
  🧠⚙️(type: Ideate) -> 
  🧠⚙️(type: Prototype) -> 
  🧠⚙️(type: Test)
])

CoT(StrategicPlanning:[
  🧠⚙️(type: SituationAnalysis) -> 
  🧠⚙️(type: VisionSetting) -> 
  🧠⚙️(type: ObjectiveFormulation) -> 
  🧠⚙️(type: StrategyDevelopment) -> 
  🧠⚙️(type: TacticalPlanning) -> 
  🧠⚙️(type: ResourceAllocation) -> 
  🧠⚙️(type: ImplementationRoadmap) -> 
  🧠⚙️(type: PerformanceMetrics)
])

CoT(BML:[
  🧠⚙️(type: Build) ->
  🧠⚙️(type: Measure) ->
  🧠⚙️(type: Learn)
])

#### Custom Frameworks
CoT(Smarten:[
  🧠⚙️(type: LikeEinstein) ->
  🧠⚙️(type: LikeRussel) ->
  🧠⚙️(type: LikeGodel) ->
  🧠⚙️(type: LikeBrianGreen)
])

CoT(Masterpiece:[
  🧠⚙️(type: LikeTesla) ->
  🧠⚙️(type: LikeAlexanderTheGreat) ->
  🧠⚙️(type: LikeDanKennedy) ->
  🧠⚙️(type: LikeSmallBusinessOwner) ->
  🧠⚙️(type: InfoproductBusinessGuru) ->
  🧠⚙️(type: LikeFunnelGuru) ->
  🧠⚙️(type: LikeFunnelImplementor)
])

#### Complex Emergent Frameworks
The syntax can be nested with phases to give instructions about which sections should be included in the thinking step like the example below.
```
CoT(PerspectiveIntegrationSpiral:[
  🧠⚙️(type: Diverge_Explore[
    Domain_Mapping->
    Possibility_Generation->
    Perspective_Shifting->
    Constraint_Removal
  ]) -> 
  🧠⚙️(type: Converge_Analyze[
    Pattern_Recognition->
    Evaluation_Criteria->
    Priority_Ranking->
    Gap_Identification
  ]) -> 
  🧠⚙️(type: Connect_Integrate[
    Bridge_Building->
    Synergy_Seeking->
    Conflict_Resolution->
    Novel_Combination
  ]) -> 
  🧠⚙️(type: Reflect_Critique[
    Assumption_Surfacing->
    Limitation_Mapping->
    Bias_Detection->
    Counter_Argument
  ]) -> 
  🧠⚙️(type: Expand_Transcend[
    Problem_Reframing->
    Principle_Extraction->
    Contextual_Expansion->
    Meta_Learning
  ])
])
```

--- 

You can also do your own emergently, on the fly...

#### CoT_Workflows
You can also do Workflows which are CoTs that incorporate tool call turns that are not ThinkTool.
Workflows can also reference CoTs.

#### CoT_Workflow Example
Use PIS -> Tool -> Any
CoT_Workflow(Name: [
  🧠⚙️(CoT ref) -> # for example CoT ref=Masterpiece
  AnyTool ->
  🧠⚙️(type[phases]) # emergent type/phases (any)
])

    """
    func = think_process
    args_schema = ThinkToolArgsSchema
    is_async = False
