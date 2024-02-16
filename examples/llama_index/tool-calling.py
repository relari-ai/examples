import json
from typing import Sequence

from llama_index.agent.openai_legacy import FnRetrieverOpenAIAgent
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.core.tools import BaseTool, FunctionTool


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def useless(a: int, b: int) -> int:
    """Toy useless function."""
    pass


def main(query, verbose: bool = True):
    tools = [
        FunctionTool.from_defaults(fn=useless, name=f"useless_{str(idx)}")
        for idx in range(10)
    ]
    tools.append(FunctionTool.from_defaults(fn=multiply, name="multiply"))
    tools.append(FunctionTool.from_defaults(fn=add, name="add"))
    obj_index = ObjectIndex.from_objects(
        tools,
        SimpleToolNodeMapping.from_objects(tools),
        VectorStoreIndex,
    )

    agent = FnRetrieverOpenAIAgent.from_retriever(
        obj_index.as_retriever(), verbose=verbose
    )
    print(agent.chat(query))


if __name__ == "__main__":
    main("What's 212 multiplied by 122?")
