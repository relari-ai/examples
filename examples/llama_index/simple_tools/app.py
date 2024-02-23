from pathlib import Path

from continuous_eval.eval.manager import eval_manager
from llama_index.agent.openai_legacy import FnRetrieverOpenAIAgent
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.core.tools import FunctionTool

from examples.llama_index.simple_tools.pipeline import pipeline

eval_manager.set_pipeline(pipeline)


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    eval_manager.log("llm", "multiply", tool_args={"a": a, "b": b})
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    eval_manager.log("llm", "multiply", tool_args={"a": a, "b": b})
    return a + b


def useless(a: int, b: int) -> int:
    """Toy useless function."""
    eval_manager.log("llm", "multiply", tool_args={"a": a, "b": b})
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
    eval_manager.start_run()
    while eval_manager.is_running():
        if eval_manager.curr_sample is None:
            break
        question = eval_manager.curr_sample["question"]
        # Retriever
        response = ask(question)
        eval_manager.log("llm", response)
        print(f"Q: {question}\nA: {response}\n")
        eval_manager.next_sample()

    eval_manager.evaluation.save(Path("results.jsonl"))
