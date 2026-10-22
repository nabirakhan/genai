import timm
import torch

from src.classify.data import NUM_CLASSES

DEFAULT_MODEL = "efficientnet_b0"


def create_model(name: str = DEFAULT_MODEL, pretrained: bool = True, num_classes: int = NUM_CLASSES,
                 drop_rate: float = 0.2) -> torch.nn.Module:
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


def gradcam_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Last conv feature map before global pooling (EfficientNet: conv_head; ResNet: layer4)."""
    for attr in ("conv_head", "layer4"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError("unknown backbone; add its last conv layer here")


def load_weights(model: torch.nn.Module, path, device="cpu") -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    return model.to(device).eval()
