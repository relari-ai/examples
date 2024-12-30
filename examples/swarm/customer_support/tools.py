import json

_ORDERS = json.load(open("examples/swarm/customer_support/data/orders.json"))


def process_refund(order_id: str):
    """Process a refund for the given order"""
    for order in _ORDERS:
        if order["id"] == order_id:
            order["status"] = "refunded"
            return f"Refund processed for order {order_id}"
    return f"Order {order_id} not found"


def look_up_order(order_id: str) -> dict | str:
    """Look up an order by its ID"""
    for order in _ORDERS:
        if order["id"] == order_id:
            return order
    return f"Order {order_id} not found"


def get_orders_dates():
    """Get all orders made on a given date"""
    return [{"date": order["date"], "id": order["id"]} for order in _ORDERS]


def escalate_to_manager(issue: str):
    """Escalate an issue to a manager"""
    return f"Issue escalated to manager: {issue}"
