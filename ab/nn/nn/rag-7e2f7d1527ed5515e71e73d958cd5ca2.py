# Auto-generated single-file for InvertibleModule
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
class torch_geometric: 
    class typing:
        WITH_PT20 = True
import numpy as np

# ---- torch_geometric.nn.models.rev_gnn.InvertibleFunction ----
class InvertibleFunction(torch.autograd.Function):
    r"""An invertible autograd function. This allows for automatic
    backpropagation in a reversible fashion so that the memory of intermediate
    results can be freed during the forward pass and be constructed on-the-fly
    during the bachward pass.

    Args:
        ctx (torch.autograd.function.InvertibleFunctionBackward):
            A context object that can be used to stash information for backward
            computation.
        fn (torch.nn.Module): The forward function.
        fn_inverse (torch.nn.Module): The inverse function to recompute the
            freed input.
        num_bwd_passes (int): Number of backward passes to retain a link
            with the output. After the last backward pass the output is
            discarded and memory is freed.
        num_inputs (int): The number of inputs to the forward function.
        *args (tuple): Inputs and weights.
    """
    def forward(ctx, fn: torch.nn.Module, fn_inverse: torch.nn.Module,
                num_bwd_passes: int, num_inputs: int, *args):
        ctx.fn = fn
        ctx.fn_inverse = fn_inverse
        ctx.weights = args[num_inputs:]
        ctx.num_bwd_passes = num_bwd_passes
        ctx.num_inputs = num_inputs
        inputs = args[:num_inputs]
        ctx.input_requires_grad = []

        with torch.no_grad():  # Make a detached copy which shares the storage:
            x = []
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    x.append(element.detach())
                    ctx.input_requires_grad.append(element.requires_grad)
                else:
                    x.append(element)
                    ctx.input_requires_grad.append(None)
            outputs = ctx.fn(*x)

        if not isinstance(outputs, tuple):
            outputs = (outputs, )

        # Detaches outputs in-place, allows discarding the intermedate result:
        detached_outputs = tuple(element.detach_() for element in outputs)

        # Clear memory of node features:
        if torch_geometric.typing.WITH_PT20:
            inputs[0].untyped_storage().resize_(0)
        else:  # pragma: no cover
            inputs[0].storage().resize_(0)

        # Store these tensor nodes for backward passes:
        ctx.inputs = [inputs] * num_bwd_passes
        ctx.outputs = [detached_outputs] * num_bwd_passes

        return detached_outputs

    def backward(ctx, *grad_outputs):
        if len(ctx.outputs) == 0:
            raise RuntimeError(
                f"Trying to perform a backward pass on the "
                f"'InvertibleFunction' for more than '{ctx.num_bwd_passes}' "
                f"times. Try raising 'num_bwd_passes'.")

        inputs = ctx.inputs.pop()
        outputs = ctx.outputs.pop()

        # Recompute input by swapping out the first argument:
        with torch.no_grad():
            inputs_inverted = ctx.fn_inverse(*(outputs + inputs[1:]))
            if len(ctx.outputs) == 0:  # Clear memory from outputs:
                for element in outputs:
                    if torch_geometric.typing.WITH_PT20:
                        element.untyped_storage().resize_(0)
                    else:  # pragma: no cover
                        element.storage().resize_(0)

            if not isinstance(inputs_inverted, tuple):
                inputs_inverted = (inputs_inverted, )

            for elem_orig, elem_inv in zip(inputs, inputs_inverted):
                if torch_geometric.typing.WITH_PT20:
                    elem_orig.untyped_storage().resize_(
                        int(np.prod(elem_orig.size())) *
                        elem_orig.element_size())
                else:  # pragma: no cover
                    elem_orig.storage().resize_(int(np.prod(elem_orig.size())))
                elem_orig.set_(elem_inv)

        # Compute gradients with grad enabled:
        with torch.set_grad_enabled(True):
            detached_inputs = []
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    detached_inputs.append(element.detach())
                else:
                    detached_inputs.append(element)
            detached_inputs = tuple(detached_inputs)
            for x, req_grad in zip(detached_inputs, ctx.input_requires_grad):
                if isinstance(x, torch.Tensor):
                    x.requires_grad = req_grad
            tmp_output = ctx.fn(*detached_inputs)

        if not isinstance(tmp_output, tuple):
            tmp_output = (tmp_output, )

        filtered_detached_inputs = tuple(
            filter(
                lambda x: x.requires_grad
                if isinstance(x, torch.Tensor) else False,
                detached_inputs,
            ))
        gradients = torch.autograd.grad(
            outputs=tmp_output,
            inputs=filtered_detached_inputs + ctx.weights,
            grad_outputs=grad_outputs,
        )

        input_gradients = []
        i = 0
        for rg in ctx.input_requires_grad:
            if rg:
                input_gradients.append(gradients[i])
                i += 1
            else:
                input_gradients.append(None)

        gradients = tuple(input_gradients) + gradients[-len(ctx.weights):]

        return (None, None, None, None) + gradients

# ---- InvertibleModule (target) ----
class InvertibleModule(torch.nn.Module):
    r"""An abstract class for implementing invertible modules.

    Args:
        disable (bool, optional): If set to :obj:`True`, will disable the usage
            of :class:`InvertibleFunction` and will execute the module without
            memory savings. (default: :obj:`False`)
        num_bwd_passes (int, optional): Number of backward passes to retain a
            link with the output. After the last backward pass the output is
            discarded and memory is freed. (default: :obj:`1`)
    """
    def __init__(self, disable: bool = False, num_bwd_passes: int = 1):
        super().__init__()
        self.disable = disable
        self.num_bwd_passes = num_bwd_passes

    def forward(self, *args):
        """"""  # noqa: D419
        return self._fn_apply(args, self._forward, self._inverse)

    def inverse(self, *args):
        return self._fn_apply(args, self._inverse, self._forward)

    def _forward(self):
        raise NotImplementedError

    def _inverse(self):
        raise NotImplementedError

    def _fn_apply(self, args, fn, fn_inverse):
        if not self.disable:
            out = InvertibleFunction.apply(
                fn,
                fn_inverse,
                self.num_bwd_passes,
                len(args),
                *args,
                *tuple(p for p in self.parameters() if p.requires_grad),
            )
        else:
            out = fn(*args)

        # If the layer only has one input, we unpack the tuple:
        if isinstance(out, tuple) and len(out) == 1:
            return out[0]

        return out

class SimpleInvertibleModule(InvertibleModule):
    def __init__(self, in_channels, out_channels, disable=False, num_bwd_passes=1):
        super().__init__(disable, num_bwd_passes)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
    
    def _forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
    def _inverse(self, x):
        x = F.relu(x, inplace=False)
        x = x / (self.bn.running_var.view(1, -1, 1, 1) + self.bn.eps).sqrt()
        x = x - self.bn.running_mean.view(1, -1, 1, 1)
        x = x * self.bn.weight.view(1, -1, 1, 1) + self.bn.bias.view(1, -1, 1, 1)
        return F.conv_transpose2d(x, self.conv.weight, padding=1)

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
        self.invertible_module = SimpleInvertibleModule(in_channels=32, out_channels=32, disable=True, num_bwd_passes=1)
        self.classifier = nn.Linear(32, self.num_classes)

    def build_features(self):
        layers = []
        layers += [
            nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.invertible_module(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.classifier(x)

    def train_setup(self, prm):
        self.to(self.device)
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
