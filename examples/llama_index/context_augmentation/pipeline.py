from continuous_eval.eval import (
    AgentModule,
    CalledTools,
    Dataset,
    Module,
    ModuleOutput,
    Pipeline,
    Tool,
)
from continuous_eval.eval.tests import GreaterOrEqualThan
from continuous_eval.metrics.generation.text import DeterministicAnswerCorrectness
from continuous_eval.metrics.tools.match import ToolSelectionAccuracy

dataset = Dataset("examples/llama_index/context_augmentation/data")

tools = [
    Tool(
        name="uber_march_10q",
        args={"input": str},
        out_type=str,
    ),
    Tool(
        name="uber_june_10q",
        args={"input": str},
        out_type=str,
    ),
    Tool(
        name="uber_sept_10q",
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
            tools=CalledTools(), ground_truths=dataset.tool_calls
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
        DeterministicAnswerCorrectness().use(
            answer=ModuleOutput(), ground_truth_answers=dataset.answer
        ),
    ],
    tests=[
        GreaterOrEqualThan(
            test_name="Answer Correctness", metric_name="rouge_l_recall", min_value=0.8
        ),
    ],
)

pipeline = Pipeline([agent, output], dataset=dataset)

print(pipeline.graph_repr())
