from swarm import Agent

from examples.swarm.customer_support.tools import (
    available_products,
    escalate_to_manager,
    get_orders,
    look_up_order,
    new_order,
    process_refund,
)

refund_agent = Agent(
    name="Refund Agent",
    model="gpt-4o-mini",
    instructions="You are a refund agent. You are responsible to help the customer with processing a refund.",
    functions=[look_up_order, process_refund, get_orders],
)

connoisseur_agent = Agent(
    name="Connoisseur Agent",
    model="gpt-4o-mini",
    instructions="You are a connoisseur agent. You are responsible to help the customer with placing a new order.",
    functions=[escalate_to_manager, new_order, available_products, get_orders],
)


def transfer_to_refund_agent():
    """Delegate an issue to a refund agent"""
    return refund_agent


def transfer_to_connoisseur_agent():
    """Delegate an issue to a connoisseur agent"""
    return connoisseur_agent


customer_service_supervisor_agent = Agent(
    name="Supervisor",
    model="gpt-4o-mini",
    instructions="""You are a supervisor agent. You are responsible for managing the customer service agent. 
    -If the customer asking for a refund, delegate the issue to the refund agent. 
    -If the customer asking to place a new order, delegate the issue to a connoisseur agent. 
    -If the customer is not satisfied, escalate the issue to a manager.""",
    functions=[
        escalate_to_manager,
        transfer_to_refund_agent,
        transfer_to_connoisseur_agent,
    ],
)
