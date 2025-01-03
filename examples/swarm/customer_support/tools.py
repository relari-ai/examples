import json
import uuid
from datetime import datetime

_ORDERS = json.load(open("examples/swarm/customer_support/data/orders.json"))
_PRODUCTS = json.load(open("examples/swarm/customer_support/data/products.json"))


def available_products():
    """Get all available products

    Returns:
        list[dict]: The list of available products
    """
    return _PRODUCTS


def new_order(first_name: str, last_name: str, email: str, items_sku: list[str]) -> str:
    """Create a new order

    Args:
        first_name (str): The first name of the customer
        last_name (str): The last name of the customer
        email (str): The email of the customer
        items_sku (list[str]): The list of SKUs (make sure it's a list)

    Returns:
        str: The ID of the new order
    """
    order_id = str(uuid.uuid4()).split("-")[0]
    items = [item for item in _PRODUCTS if item["sku"] in items_sku]
    if len(items) != len(items_sku):
        return "Error: Some items are not available"
    _ORDERS.append(
        {
            "id": order_id,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "items": items,
            "status": "pending",
            "date": datetime.now().isoformat(),
        }
    )
    return order_id


def process_refund(order_id: str) -> str:
    """Process a refund for the given order

    Args:
        order_id (str): The ID of the order to process a refund for

    Returns:
        str: Message indicating the status of the refund
    """
    for order in _ORDERS:
        if order["id"] == order_id:
            order["status"] = "refunded"
            return f"Refund processed for order {order_id}"
    return f"Order {order_id} not found"


def look_up_order(order_id: str) -> dict | str:
    """Look up an order by its ID

    Args:
        order_id (str): The ID of the order to look up

    Returns:
        dict | str: The order if found, otherwise an error message
    """
    for order in _ORDERS:
        if order["id"] == order_id:
            return order
    return f"Order {order_id} not found"


def get_orders():
    """Get all orders"""
    return _ORDERS


def escalate_to_manager(issue: str):
    """Escalate an issue to a manager

    Args:
        issue (str): The issue to escalate

    Returns:
        str: Message from the manager
    """
    return f"Issue escalated to manager: {issue}"
