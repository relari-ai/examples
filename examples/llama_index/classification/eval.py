from pathlib import Path

from continuous_eval.eval.manager import eval_manager
from examples.llama_index.classification.pipeline import pipeline

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
    print("Done")