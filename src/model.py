"""Image encoder factory and FC classifier builder."""

import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = ['resnet18', 'densenet121', 'efficientnet_b0', 'mobilenet_v2', 'inceptionv3']

_WEIGHTS_MAP = {
    'resnet18':        models.ResNet18_Weights.IMAGENET1K_V1,
    'densenet121':     models.DenseNet121_Weights.IMAGENET1K_V1,
    'efficientnet_b0': models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    'mobilenet_v2':    models.MobileNet_V2_Weights.IMAGENET1K_V1,
    'inceptionv3':     models.Inception_V3_Weights.IMAGENET1K_V1,
}


def get_model(model_name, num_classes, pt=True):
    """Return a torchvision backbone with a re-initialized classification head.

    Returns (model, img_size). InceptionV3 uses img_size=299; others use 224.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {SUPPORTED_MODELS}")

    weights = _WEIGHTS_MAP[model_name] if pt else None

    if model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        new_head = nn.Linear(model.fc.in_features, num_classes)
        _init_head(new_head)
        model.fc = new_head
        img_size = 224

    elif model_name == 'densenet121':
        model = models.densenet121(weights=weights)
        new_head = nn.Linear(model.classifier.in_features, num_classes)
        _init_head(new_head)
        model.classifier = new_head
        img_size = 224

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        new_head = nn.Linear(model.classifier[1].in_features, num_classes)
        _init_head(new_head)
        model.classifier[1] = new_head
        img_size = 224

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=weights)
        new_head = nn.Linear(model.classifier[1].in_features, num_classes)
        _init_head(new_head)
        model.classifier[1] = new_head
        img_size = 224

    elif model_name == 'inceptionv3':
        # aux_logits=False to get a single tensor output
        model = models.inception_v3(weights=weights, aux_logits=False)
        new_head = nn.Linear(model.fc.in_features, num_classes)
        _init_head(new_head)
        model.fc = new_head
        img_size = 299

    return model, img_size


def _init_head(layer):
    """Kaiming-uniform init for the classification head (bias zeroed)."""
    nn.init.kaiming_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def create_fc_model(input_size, n_layers=2, first_hidden=128, hidden_sizes=None,
                    use_dropout=False, dropout_rate=0.3):
    """Build a variable-depth FC classifier. If n_layers=1, returns a single Linear to 2 classes."""
    if hidden_sizes is None:
        hidden_sizes = []

    if n_layers == 1:
        return nn.Linear(input_size, 2)

    layers = [nn.Linear(input_size, first_hidden), nn.ReLU()]
    if use_dropout:
        layers.append(nn.Dropout(dropout_rate))

    current_size = first_hidden
    for size in hidden_sizes:
        layers.extend([nn.Linear(current_size, size), nn.ReLU()])
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        current_size = size

    layers.append(nn.Linear(current_size, 2))
    return nn.Sequential(*layers)
