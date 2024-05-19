from pathlib import Path

from continuous_eval.eval.logger import PipelineLogger
from continuous_eval.eval.runner import EvaluationRunner
from examples.llama_index.classification.pipeline import pipeline

if __name__ == "__main__":
    output_dir = Path("output")

    pipelog = PipelineLogger(pipeline=pipeline)
    pipelog.load(output_dir / "llamaindex_classification.jsonl")

    evalrunner = EvaluationRunner(pipeline)
    metrics = evalrunner.evaluate(pipelog)
    metrics.save(output_dir / "llamaindex_classification_metrics.json")
    print(metrics.aggregate())