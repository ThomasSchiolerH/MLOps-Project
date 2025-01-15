import torch
import typer
from data_solution import corrupt_mnist
from model import PricePredictionModel
from sklearn.ensemble import RandomForestRegressor

app = typer.Typer()


@app.command()
def train(n: int) -> None:
    print("Training day and night")

    model = RandomForestRegressor()
    train_set, _ = corrupt_mnist()


@app.command()
def evaluate(model_checkpoint: str) -> None:

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = corrupt_mnist()


if __name__ == "__main__":
    app()