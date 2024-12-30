import json
from pathlib import Path

from continuous_eval.eval.logger import PipelineLogger
from continuous_eval.eval.runner import EvaluationRunner

from examples.common import print_metric_results, print_test_results
from examples.swarm.customer_support.pipeline import pipeline
from continuous_eval.eval import Pipeline

def swarm2log(pipeline: Pipeline, log_path: str) -> dict:
    # First load the log
    with open(log_path, "r") as f:
        log_data = json.load(f)
    log = PipelineLogger(pipeline=pipeline)
    # Then for every entry (uid, messages) in the log
    for uid, messages in log_data.items():
        for m in messages:
            # Only for the messages of an agent, log the tool calls
            if m["role"] == "assistant" and m["tool_calls"]:
                for t in m["tool_calls"]:
                    log.log(
                        uid=uid,
                        module=m["sender"],
                        value=t["function"]["name"],
                        tool_args=json.loads(t["function"]["arguments"]),
                    )
    return log


if __name__ == "__main__":
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    log = swarm2log(pipeline, output_dir / "swarm_customer_support_agent.jsonl")
    log.save(output_dir / "swarm_customer_support_agent_parsed_log.jsonl")

    evalrunner = EvaluationRunner(pipeline)
    metrics = evalrunner.evaluate(log)
    metrics.save(output_dir / "swarm_customer_support_agent_metrics.json")
    tests = evalrunner.test(metrics)

    print_metric_results(metrics)
    print_test_results(tests)
