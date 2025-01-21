from src.AVM.model import PricePredictionModel
from src.AVM.data import FEATURE_COLUMNS

# Test model loading
def test_model_loading():
    model = PricePredictionModel(in_features=len(FEATURE_COLUMNS))
    assert model is not None, "Model failed to initialize"
    
    