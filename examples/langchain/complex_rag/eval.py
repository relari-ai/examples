from pathlib import Path

from continuous_eval.eval.logger import PipelineLogger
from continuous_eval.eval.runner import EvaluationRunner
from examples.common import print_metric_results, print_test_results
from examples.langchain.complex_rag.pipeline import pipeline

if __name__ == "__main__":
    output_dir = Path("output")

    pipelog = PipelineLogger(pipeline=pipeline)
    pipelog.load(output_dir / "langchain_complex_rag.jsonl")

    # Run the evaluation...
    evalrunner = EvaluationRunner(pipeline)
    metrics = evalrunner.evaluate(pipelog)
    metrics.save(output_dir / "langchain_complex_rag_metrics.json")
    # ...or you can load from file
    # metrics = MetricsResults(pipeline)
    # metrics.load(output_dir / "langchain_complex_rag_metrics.json")
    tests = evalrunner.test(metrics)
    
    print_metric_results(metrics)
    print_test_results(tests)
