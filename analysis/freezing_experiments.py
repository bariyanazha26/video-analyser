from ultralytics import YOLO

# Load pretrained YOLOv8L
model = YOLO("yolov8l.pt")

def configure_freezing(model, strategy):
    backbone = range(0, 10)
    head = [22]

    if strategy == "backbone":
        frozen_layers = set(backbone)
    elif strategy == "head":
        frozen_layers = set(head)
    elif strategy == "both":
        frozen_layers = set(backbone) | set(head)
    elif strategy == "none":
        frozen_layers = set()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    for i, layer in enumerate(model.model.model):
        freeze = i in frozen_layers

        for parameter in layer.parameters():
            parameter.requires_grad = not freeze

    print(f"\nFreezing strategy: {strategy.upper()}")

    for i, layer in enumerate(model.model.model):
        parameters = list(layer.parameters())
        if not parameters:
            status = "NO PARAMETERS"
        elif all(not p.requires_grad for p in parameters):
            status = "FROZEN"
        else:
            status = "TRAINABLE"

        print(f"Layer {i:2d}: {status}")


# Test configuration without training
configure_freezing(model, "both")