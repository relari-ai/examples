from continuous_eval.eval import Dataset

data = [
    {
        "uid": "ref30",
        "question": "I want to return my order made on Dec 30",
        "supervisor_tool_calls": [
            {"name": "transfer_to_refund_agent", "kwargs": {}},
        ],
        "refund_tool_calls": [
            {"name": "get_orders_dates", "kwargs": {}},
            {"name": "process_refund", "kwargs": {"order_id": "b4a402a1"}},
        ],
    }
]

dataset = Dataset.from_data(data)
dataset.save("examples/swarm/customer_support/data/dataset.jsonl")
