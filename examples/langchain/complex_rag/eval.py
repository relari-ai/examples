from pathlib import Path

from continuous_eval.eval.manager import eval_manager
from examples.langchain.complex_rag.pipeline import pipeline

if __name__ == "__main__":
    eval_manager.set_pipeline(pipeline)

    # Evaluation
    # eval_manager.load_results(Path("results.jsonl"))
    # eval_manager.run_eval()
    # eval_manager.save_eval_results(Path("eval_results.json"))

    # Tests
    eval_manager.load_eval_results(Path("eval_results.json"))
    eval_manager.run_tests()
    eval_manager.save_test_results(Path("test_results.json"))

    # eval_manager.load_test_results(Path("test_results.json"))
    for test in eval_manager.test_results:
        print(f"{test}: {eval_manager.test_results[test]}")
    print("Done")