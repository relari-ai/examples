from pathlib import Path

from continuous_eval.eval.logger import PipelineLogger
from llama_index.agent.openai_legacy import FnRetrieverOpenAIAgent
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.core.tools import FunctionTool

from examples.llama_index.simple_tools.pipeline import pipeline

pipelog = PipelineLogger(pipeline=pipeline)
curr_uid = None


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    pipelog.log(uid=curr_uid, module="llm", value="multiply", tool_args={"a": a, "b": b})
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    pipelog.log(uid=curr_uid, module="llm", value="add", tool_args={"a": a, "b": b})
    return a + b


def useless(a: int, b: int) -> int:
    """Toy useless function."""
    pipelog.log(uid=curr_uid, module="llm", value="useless", tool_args={"a": a, "b": b})
    return a


def ask(query, verbose: bool = True):
    tools = [
        FunctionTool.from_defaults(fn=useless, name=f"useless"),
        FunctionTool.from_defaults(fn=multiply, name="multiply"),
        FunctionTool.from_defaults(fn=add, name="add"),
    ]
    obj_index = ObjectIndex.from_objects(
        tools,
        SimpleToolNodeMapping.from_objects(tools),
        VectorStoreIndex,
    )
    agent = FnRetrieverOpenAIAgent.from_retriever(
        obj_index.as_retriever(), verbose=verbose
    )
    response = agent.chat(query)
    return response.response


if __name__ == "__main__":
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for datum in pipelog.pipeline.dataset.data:
        curr_uid = datum["uid"] # set the global variable
        # Retriever
        response = ask(datum["question"])
        pipelog.log(uid=curr_uid, module="llm", value=response)
        print(f"Q: {datum['question']}\nA: {response}\n")

    pipelog.save(output_dir/"llamaindex_tools.jsonl")