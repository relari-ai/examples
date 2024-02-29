from pathlib import Path

from relari.eval.manager import eval_manager
from examples.langchain.simple_rag.pipeline import pipeline
from relari import RelariClient

if __name__ == "__main__":
    eval_manager.set_pipeline(pipeline)
    eval_manager.evaluation.load(Path("results.jsonl"))

    client = RelariClient()
    client.start_remote_evaluation()