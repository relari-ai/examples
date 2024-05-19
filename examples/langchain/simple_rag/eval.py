from pathlib import Path

from continuous_eval.eval.logger import PipelineLogger
from continuous_eval.eval.result_types import MetricsResults
from continuous_eval.eval.runner import EvaluationRunner

from examples.langchain.simple_rag.pipeline import pipeline

if __name__ == "__main__":
    output_dir = Path("output")

    pipelog = PipelineLogger(pipeline=pipeline)
    pipelog.load(output_dir / "langchain_simple_rag.jsonl")

    # Run the evaluation...
    evalrunner = EvaluationRunner(pipeline)
    metrics = evalrunner.evaluate(pipelog)
    metrics.save(output_dir / "langchain_simple_rag_metrics.json")
    # ...or you can load from file
    # metrics = MetricsResults(pipeline)
    # metrics.load(output_dir / "langchain_simple_rag_metrics.json")
    print(metrics.aggregate())

    print("\nTests results:")
    tests = evalrunner.test(metrics)
    for module_name, test_results in tests.results.items():
        print(f"{module_name}")
        for test_name in test_results:
            print(f" - {test_name}: {'PASS' if test_results[test_name] else 'FAIL'}")
