from continuous_eval.eval import (
    CalledTools,
    Dataset,
    Module,
    Pipeline,
    Tool,
)
from continuous_eval.metrics.tools.match import ToolSelectionAccuracy


dataset = Dataset("examples/swarm/customer_support/data/dataset.jsonl")

supervisor = Module(
    name="Supervisor",
    input=dataset.question,
    tools=[
        Tool(name="escalate_to_manager"),
        Tool(name="transfer_to_refund_agent"),
        Tool(name="transfer_to_connoisseur_agent"),
    ],
    eval=[
        ToolSelectionAccuracy(order_sensitive=True).use(
            tools=CalledTools(), ground_truths=dataset.supervisor_tool_calls
        ),
    ],
)

refund_agent = Module(
    name="Refund Agent",
    input=supervisor,
    tools=[
        Tool(name="process_refund", args={"order_id": str}),
        Tool(name="look_up_order", args={"order_id": str}),
        Tool(name="get_orders"),
    ],
    eval=[
        ToolSelectionAccuracy().use(
            tools=CalledTools(), ground_truths=dataset.refund_tool_calls
        ),
    ],
)

connoisseur_agent = Module(
    name="Connoisseur Agent",
    input=supervisor,
    tools=[
        Tool(name="get_orders"),
        Tool(name="available_products"),
        Tool(
            name="new_order",
            args={
                "first_name": str,
                "last_name": str,
                "email": str,
                "items_sku": list[str],
            },
        ),
    ],
    eval=[
        ToolSelectionAccuracy().use(
            tools=CalledTools(), ground_truths=dataset.connoisseur_tool_calls
        ),
    ],
)

pipeline = Pipeline([supervisor, refund_agent, connoisseur_agent], dataset=dataset)

if __name__ == "__main__":
    print(pipeline.graph_repr())
