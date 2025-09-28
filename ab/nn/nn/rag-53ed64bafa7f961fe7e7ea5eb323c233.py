# Auto-generated single-file for FXModel
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn

# ---- ultralytics.utils.torch_utils.copy_attr ----
def copy_attr(a, b, include=(), exclude=()):
    """
    Copy attributes from object 'b' to object 'a', with options to include/exclude certain attributes.

    Args:
        a (Any): Destination object to copy attributes to.
        b (Any): Source object to copy attributes from.
        include (tuple, optional): Attributes to include. If empty, all attributes are included.
        exclude (tuple, optional): Attributes to exclude.
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)

# ---- FXModel (target) ----
class FXModel(nn.Module):
    """
    A custom model class for torch.fx compatibility.

    This class extends `torch.nn.Module` and is designed to ensure compatibility with torch.fx for tracing and graph
    manipulation. It copies attributes from an existing model and explicitly sets the model attribute to ensure proper
    copying.

    Attributes:
        model (nn.Module): The original model's layers.
    """

    def __init__(self, model):
        """
        Initialize the FXModel.

        Args:
            model (nn.Module): The original model to wrap for torch.fx compatibility.
        """
        super().__init__()
        copy_attr(self, model)
        # Explicitly set `model` since `copy_attr` somehow does not copy it.
        self.model = model.model

    def forward(self, x):
        """
        Forward pass through the model.

        This method performs the forward pass through the model, handling the dependencies between layers and saving
        intermediate outputs.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            (torch.Tensor): The output tensor from the model.
        """
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                # from earlier layers
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)  # run
            y.append(x)  # save output
        return x

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.image_size = in_shape[2]
        self.num_classes = out_shape[0]
        self.learning_rate = prm['lr']
        self.momentum = prm['momentum']
        self.features = self.build_features()
        self.fx_model = self.build_fx_model()
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)

    def build_fx_model(self):
        class SimpleLayer:
            def __init__(self, layer, f=-1):
                self.layer = layer
                self.f = f
            
            def __call__(self, x):
                return self.layer(x)
        
        model_layers = [
            SimpleLayer(nn.Conv2d(32, 32, 3, padding=1), f=-1),
            SimpleLayer(nn.BatchNorm2d(32), f=0),
            SimpleLayer(nn.ReLU(), f=1),
        ]
        
        class ModelWrapper:
            def __init__(self, layers):
                self.model = layers
        
        fx_model = FXModel(ModelWrapper(model_layers))
        return fx_model

    def forward(self, x):
        x = self.features(x)
        x = self.fx_model(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
        self.fx_model.to(self.device)
        for layer in self.fx_model.model:
            layer.layer.to(self.device)
        self.criteria = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=5e-4)

    def learn(self, data_roll):
        self.train()
        for batch_idx, (data, target) in enumerate(data_roll):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criteria(output, target)
            loss.backward()
            self.optimizer.step()
