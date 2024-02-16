from examples.eval import Module, AgentModule, Pipeline, Tool, Dataset

dataset = Dataset("uber")

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
    name="rag",
    input=dataset.question,
    output=str,
    expected_output=dataset.answer,
    expected_tool_calls=dataset.tool_calls,
    tools=tools,
)

output = Module(
    name="answer",
    input=agent,
    output=str,
    expected_output=dataset.answer,
)

pipeline = Pipeline([agent, output], dataset=dataset)

