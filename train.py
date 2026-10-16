"""Train the demonstrator and save patient-disjoint synthetic evaluation metrics."""
from model import train

if __name__ == "__main__":
    import json
    print(json.dumps(train(), indent=2))
