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
        ToolSelectionAccuracy().use(
            tools=CalledTools(), ground_truths=dataset.supervisor_tool_calls
        ),
    ],
)

refund_agent = Module(
    name="Refund Agent",
    input=supervisor,
    eval=[
        ToolSelectionAccuracy().use(
            tools=CalledTools(), ground_truths=dataset.refund_tool_calls
        ),
    ],
    tools=[
        Tool(name="process_refund", args={"order_id": str}),
        Tool(name="look_up_order", args={"order_id": str}),
        Tool(name="get_orders_dates"),
    ],
)

connoisseur_agent = Module(
    name="Connoisseur Agent",
    input=supervisor,
)

pipeline = Pipeline([supervisor, refund_agent, connoisseur_agent], dataset=dataset)

if __name__ == "__main__":
    print(pipeline.graph_repr())
