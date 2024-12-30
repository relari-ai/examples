import random
import uuid
import json
from datetime import datetime, timedelta

# Possible data for generation
first_names = ["John", "Jane", "Alex", "Emily", "Chris", "Sara", "Michael", "Laura", "David", "Sophia"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Martinez", "Lopez"]
statuses = ["processing", "shipped", "delivered", "refunded"]
items_catalog = [
    {"sku": "1001", "name": "Wireless Mouse", "price": 29.99},
    {"sku": "1002", "name": "Keyboard", "price": 49.99},
    {"sku": "1003", "name": "Monitor", "price": 199.99},
    {"sku": "1004", "name": "USB Cable", "price": 9.99},
    {"sku": "1005", "name": "Laptop Stand", "price": 39.99},
    {"sku": "1006", "name": "Webcam", "price": 89.99},
    {"sku": "1007", "name": "Headphones", "price": 59.99},
    {"sku": "1008", "name": "External SSD", "price": 149.99},
    {"sku": "1009", "name": "Phone Case", "price": 19.99},
    {"sku": "1010", "name": "Charger", "price": 24.99},
]


def generate_mock_data(num_samples):
    start_date = datetime.now()
    data = []
    for i in range(num_samples):
        name, last_name = random.choice(first_names), random.choice(last_names)
        customer = {
            "id": str(uuid.uuid4()).split("-")[0],
            "first_name": name,
            "last_name": last_name,
            "email": f"{name.lower()}.{last_name.lower()}@example.com",
            "status": random.choice(statuses),
            "items": random.sample(items_catalog, random.randint(1, 5)),
            "date": (start_date + timedelta(days=i)).isoformat(),
        }
        data.append(customer)
    return data



if __name__ == "__main__":
    data = generate_mock_data(10)
    with open("examples/swarm/simple/data.json", "w") as f:
        json.dump(data, f, indent=None)
