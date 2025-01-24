from locust import HttpUser, task, between
import random


class FastAPIUser(HttpUser):
    wait_time = between(1, 3)  # Simulate user wait time between requests

    @task(1)
    def health_check(self):
        """Test the health check endpoint."""
        self.client.get("/")

    @task(3)
    def predict(self):
        """Test the predict endpoint with random data."""
        example_data = {
            "FLOOR": "1",
            "CONSTRUCTION_YEAR": random.randint(1950, 2025),
            "REBUILDING_YEAR": 0,
            "DISTANCE_LAKE": random.uniform(1000, 5000),
            "DISTANCE_HARBOUR": random.uniform(1000, 5000),
            "DISTANCE_COAST": random.uniform(1000, 5000),
            "HAS_ELEVATOR": random.choice(["true", "false"]),
            "AREA_TINGLYST": random.randint(50, 150),
            "AREA_RESIDENTIAL": random.randint(50, 150),
            "AREA_OTHER": random.uniform(0, 20),
            "AREA_COMMON_ACCESS_SHARE": random.uniform(0, 10),
            "AREA_CLOSED_COVER_OUTHOUSE": random.uniform(0, 10),
            "AREA_OPEN_BALCONY_ROOFTOP": random.uniform(0, 10),
            "MUNICIPALITY_CODE": random.randint(100, 999),
            "ZIP_CODE": random.randint(1000, 9999),
            "TRADE_YEAR": random.randint(2000, 2025),
            "TRADE_MONTH": random.randint(1, 12),
            "TRADE_DAY": random.randint(1, 28),
            "NUMBER_ROOMS": random.randint(1, 5),
            "TRADE_DATE": "2024-06-30",
        }
        self.client.post("/predict", json=example_data)
