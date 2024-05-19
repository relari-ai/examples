from pathlib import Path

from continuous_eval.eval.logger import PipelineLogger
from continuous_eval.eval.runner import EvaluationRunner

from examples.llama_index.simple_tools.pipeline import pipeline

if __name__ == "__main__":
    output_dir = Path("output")

    pipelog = PipelineLogger(pipeline=pipeline)
    pipelog.load(output_dir / "llamaindex_tools.jsonl")

    evalrunner = EvaluationRunner(pipeline)
    metrics = evalrunner.evaluate(pipelog)
    metrics.save(output_dir / "llamaindex_tools_metrics.json")
    print(metrics.aggregate())

    print("\nTests results:")
    tests = evalrunner.test(metrics)
    for module_name, test_results in tests.results.items():
        print(f"{module_name}")
        for test_name in test_results:
            print(f" - {test_name}: {'PASS' if test_results[test_name] else 'FAIL'}")
