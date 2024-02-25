from continuous_eval.eval import Module, AgentModule, Pipeline, Tool, Dataset, ModuleOutput
from continuous_eval.metrics.generation.text import DeterministicAnswerCorrectness
from continuous_eval.metrics.tools.match import ToolSelectionAccuracy
from continuous_eval.eval.tests import GreaterOrEqualThan

dataset = Dataset("examples/llama_index/context_augmentation/data")

tools = [
    Tool(
        name="march",
        args={"input": str},
        out_type=str,
    ),
    Tool(
        name="june",
        args={"input": str},
        out_type=str,
    ),
    Tool(
        name="sept",
        args={"input": str},
        out_type=str,
    ),
]

agent = AgentModule(
    name="retriever_agent",
    input=dataset.question,
    output=str,
    eval=[
        ToolSelectionAccuracy().use(
            tools=tools, ground_truths=dataset.tool_calls
        ),
    ],
    tests=[
        GreaterOrEqualThan(
            test_name="Tool Selection Accuracy", metric_name="score", min_value=0.8
        ),
    ],
)

output = Module(
    name="answer",
    input=agent,
    output=str,
    eval=[
        DeterministicAnswerCorrectness().use(answer=ModuleOutput(), ground_truth=dataset.answer),
    ],
    tests=[
        GreaterOrEqualThan(
            test_name="Answer Correctness", metric_name="rouge_l_recall", min_value=0.8
        ),
    ],)

pipeline = Pipeline([agent, output], dataset=dataset)

print(pipeline.graph_repr())
