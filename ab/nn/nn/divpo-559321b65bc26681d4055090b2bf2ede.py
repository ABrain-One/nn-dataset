import torch
import torch.nn as nn

def supported_hyperparameters():
    return {'lr', 'momentum'}

class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
        self.in_channels = in_shape[1]
        self.num_classes = int(out_shape[0])
        self.learning_rate = prm.get('lr', 0.01)
        self.momentum = prm.get('momentum', 0.9)
        
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Linear(128 * 6 * 6, self.num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=prm.get('weight_decay', 0.0005)
        )
    
    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

# Auto-generated single-file for EfficientNetB0
# Dependencies are created following the export pattern:
#   - after defining class, immediately downstairs lambda-defined no-op (for layers/functions)
#   - then new type classes (blocks, helpers) after the target class
#   - then the compound build() method (inside target class ONLY)
#   - then the top-level handler that combines the components into a working Net
#   - then the Net class (target) and the handler fall outside the top-level scope
#   - ALL method bodies are FRESHLY generated (do NOT repurpose across classes)
#   - ALL method signatures are (shallow) no-args       -> define@          (lambda) target.name
#                                                      create@           lambda: target.type(name=uniq_name, *args, **kwargs)
#                                                      after define@      lambda: target(name=target.name, *args, **kwargs)
#   - result is CHECK so that ALL names are UNTIL now     unique (NO_NAME in response)
#     - helper classes have name=existing-name (e.g.    SELU = torch.nn.SELU
#     - the Net class has name="Net" (target)
#     - NO net.__name__ = Net (so that UNIQUE name is net.uniq_name)
#   - target is CHOSEN randomly from the list of         possible: EfficientNetB0, InvertedResidual, InvertedResidualV2, SELU
#   - target MUST NOT appear in the list {               EfficientNetB0, InvertedResidual, InvertedResidualV2, SELU
#       'effnetb0', 'invertedres', 'invertedres',
#       'selu'
#     } unless the round contains a primary training                       (no extra puzzles)
#     round 1: primary ONLY (efficientnetb0)
#     round 2: secondary (invertedres, invertedres)
#     round 3: tertiary (selu)
#     round 4 and later: mixed bag of categories
#     If the primary target appears in the list, it MUST be the first choice.
#     If the secondary target also appears in the list, it MUST be the second choice after the primary.
#     tertiary (selu) category MUST NOT appear in the list until round 3 or later.
#     The NO_NAME restriction applies to all class definitions and names derived from the target class.
#     The generated class name for the target class will be net.uniq_name = "Net" + an original suffix.
#     net.uniq_name MUST NOT be one of the known class names: ['Net', 'Conv', 'Linear', 'Sequential', 'Module', 'Tensor']
#     The generated class name MUST NOT be one of the known class names: ['EfficientNet', 'InvertedResidual', 'InvertedResidualV2', 'SELU']
#     The generated class name for each block/function type (after the target) MUST adhere to the pattern: net[PROP_NAME]X = lambda: nn[PROP_NAME] + OriginalSuffix
#     For PROP_NAME, see the list of known property names: ['Conv', 'Conv2d', 'Linear', 'BatchNorm', 'BatchNorm2d', 'ReLU', 'ReLU6', 'AdaptiveAvgPool2d']
#     OriginalSuffix is one of: '_', '_MOD', 'Block', 'V2Block' (for blocks) or None (for functions)
#     For example, if the target is EfficientNetB0, the NO_NAME generated class name could be: net.uniq_name = "EffNetB0" + "_MOD"
#     The target class name is: EfficientNetB0 (so the generated class would be based on it)
#     The NO_NAME generated class would have the signature near to: class EfficientNetB0_MOD(nn.Module):
#     The block class (after the target) would have the signature near to: class EfficientNetB0Block(InvertedResidualV2):
#     The function would have the signature near to: def effnetb0_mod(in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> nn.Module:
#     The NO_NAME generated class would NOT be copy-constructs (keep UNIQUE).
#     The NO_NAME class would NOT have constructor arguments (target has NO constructor).
#     The NO_NAME class would NOT reference the target class by name (so you can't see it in the code).
#     The generated class would NOT copy the target's forward() or any other methods.
#     The generated class would NOT have any imports except from torch.nn (consider torch.nn as FULLY provided)
#     The generated class would NOT have any from torch.nn import ... statements (no unpacking or specific imports)
#     The generated class would NOT have any from ... import ... or import ... statements (no additional imports)
#     The generated class would ONLY have ONE import statement near the beginning: import torch.nn as nn
#     The generated class would ONLY use properties from the provided PROP_NAME list.
#     The generated class would ONLY use methods and properties from the provided PROP_DICT dictionary.
#     The generated class would ONLY use no-argument lambda functions as placeholders.
#     The generated class would ONLY use the provided SELU, InvertedResidualV2, InvertedResidual types (if at all).
#     The generated class would ONLY use the provided prm, net dictionary for method calls and property accesses.
#