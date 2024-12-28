from continuous_eval.eval.result_types import MetricsResults, TestResults


def print_metric_results(metrics: MetricsResults) -> None:
    agg = metrics.aggregate()
    print("# Metrics results:")
    for module_name, module_results in agg.items():
        print(f"> {module_name}")
        for metric_name, metric_results in module_results.items():
            print(f" - {metric_name}: {metric_results}")
    print("\n")


def print_test_results(tests: TestResults) -> None:
    print("# Tests results:")
    for module_name, test_results in tests.results.items():
        print(f"> {module_name}")
        for test_name in test_results:
            print(f" - {test_name}: {'PASS' if test_results[test_name] else 'FAIL'}")
    print("\n")