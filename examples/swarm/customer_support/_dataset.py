from continuous_eval.eval import Dataset

data = [
    {
        "uid": "ref30",
        "question": "I want to return my order made on Dec 30",
        "supervisor_tool_calls": [
            {"name": "transfer_to_refund_agent", "kwargs": {}},
        ],
        "refund_tool_calls": [
            {"name": "get_orders", "kwargs": {}},
            {"name": "process_refund", "kwargs": {"order_id": "b4a402a1"}},
        ],
        "connoisseur_tool_calls": [],
    },
    {
        "uid": "new",
        "question": "I want to order a new laptop stand",
        "supervisor_tool_calls": [
            {"name": "transfer_to_connoisseur_agent", "kwargs": {}},
        ],
        "refund_tool_calls": [],
        "connoisseur_tool_calls": [
            {
                "name": "new_order",
                "kwargs": {
                    "first_name": "John",
                    "last_name": "Doe",
                    "email": "john.doe@example.com",
                    "items_sku": ["1005"],
                },
            },
        ],
    },
]

dataset = Dataset.from_data(data)
dataset.save("examples/swarm/customer_support/data/dataset.jsonl")
