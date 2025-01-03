import json
from pathlib import Path

from swarm import Swarm
from tqdm import tqdm

from examples.swarm.customer_support.agents import customer_service_supervisor_agent
from examples.swarm.customer_support.pipeline import pipeline

if __name__ == "__main__":
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    log = {}

    print("Running pipeline...")
    client = Swarm()
    for datum in tqdm(pipeline.dataset.data):
        response = client.run(
            agent=customer_service_supervisor_agent,
            messages=[
                {
                    "role": "system",
                    "content": "You are a customer service agent. You are responsible to help John Doe (email: john.doe@example.com) with their issue.",
                },
                {"role": "user", "content": datum["question"]},
            ],
            debug=True,
        )
        log[datum["uid"]] = response.messages

    out_fname = output_dir / "swarm_customer_support_agent.jsonl"
    with open(out_fname, "w") as f:
        json.dump(log, f, indent=2)
    print("Pipeline run completed.")
    print(f"Results saved to {out_fname}")
