"""Neural network architectures for Human Activity Recognition."""

from .cnn_lstm import CNNLSTMClassifier, build_model_from_config

__all__ = ["CNNLSTMClassifier", "build_model_from_config"]
