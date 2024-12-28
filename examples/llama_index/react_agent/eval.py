from pathlib import Path

from continuous_eval.eval.logger import PipelineLogger
from continuous_eval.eval.runner import EvaluationRunner
from examples.common import print_metric_results, print_test_results
from examples.llama_index.react_agent.pipeline import pipeline

if __name__ == "__main__":
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    pipelog = PipelineLogger(pipeline=pipeline)
    pipelog.load(output_dir / "llamaindex_react_agent.jsonl")

    evalrunner = EvaluationRunner(pipeline)
    metrics = evalrunner.evaluate(pipelog)
    metrics.save(output_dir / "llamaindex_react_agent_metrics.json")
    tests = evalrunner.test(metrics)
    
    print_metric_results(metrics)
    print_test_results(tests)
