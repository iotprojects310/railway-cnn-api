import torch
from src.model import ConvAutoencoder

def test_model_forward_shape():
    model = ConvAutoencoder(in_channels=3)
    x = torch.randn(4, 3, 128)
    y = model(x)
    assert y.shape == x.shape
