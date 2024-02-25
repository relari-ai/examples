from continuous_eval.eval import (
    Tool,
    AgentModule,
    Pipeline,
    Dataset,
    ModuleOutput,
    CalledTools,
)
from continuous_eval.metrics.generation.text import DeterministicAnswerCorrectness
from continuous_eval.metrics.tools.match import ToolSelectionAccuracy
from continuous_eval.eval.tests import GreaterOrEqualThan

dataset = Dataset("examples/llama_index/simple_tools/data")

add = Tool(
    name="add",
    args={"a": int, "b": int},
    out_type=int,
    description="Add two integers and returns the result integer",
)

multiply = Tool(
    name="multiply",
    args={"a": int, "b": int},
    out_type=int,
    description="Multiply two integers and returns the result integer",
)

useless = Tool(
    name="useless",
    args={"a": int, "b": int},
    out_type=int,
    description="Toy useless function.",
)

llm = AgentModule(
    name="llm",
    input=dataset.question,
    output=str,
    tools=[add, multiply, useless],
    eval=[
        DeterministicAnswerCorrectness().use(
            answer=ModuleOutput(), ground_truth_answers=dataset.answers
        ),
        ToolSelectionAccuracy().use(
            tools=CalledTools(), ground_truths=dataset.tools
        ),
    ],
    tests=[
        # GreaterOrEqualThan(
        #     test_name="Readability", metric_name="flesch_reading_ease", min_value=20.0
        # ),
        # GreaterOrEqualThan(
        #     test_name="Answer Correctness", metric_name="rouge_l_f1", min_value=0.8
        # ),
    ],
)

pipeline = Pipeline([llm], dataset=dataset)

print(pipeline.graph_repr())
