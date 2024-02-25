from pathlib import Path

from continuous_eval.eval.manager import eval_manager
from examples.llama_index.simple_tools.pipeline import pipeline

if __name__ == "__main__":
    eval_manager.set_pipeline(pipeline)

    # Evaluation
    eval_manager.evaluation.load(Path("results.jsonl"))
    eval_manager.run_metrics()
    eval_manager.metrics.save(Path("metrics_results.json"))

    # Tests
    eval_manager.metrics.load(Path("metrics_results.json"))
    agg = eval_manager.metrics.aggregate()
    print(agg)
    eval_manager.run_tests()
    eval_manager.tests.save(Path("test_results.json"))

    eval_manager.tests.load(Path("test_results.json"))
    for module_name, test_results in eval_manager.tests.results.items():
        print(f"{module_name}")
        for test_name in test_results:
            print(f" - {test_name}: {test_results[test_name]}")
    print("Done")