"""Teacher/student CNNs and physically compact filter pruning."""
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, channels=(8, 16), hidden=1280):
        super().__init__()
        self.channels, self.hidden = tuple(channels), hidden
        self.conv1 = nn.Conv2d(1, channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.fc1 = nn.Linear(channels[1]*7*7, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = torch.nn.functional.max_pool2d(self.conv1(x).relu(), 2)
        x = torch.nn.functional.max_pool2d(self.conv2(x).relu(), 2)
        return self.fc2(self.fc1(x.flatten(1)).relu())


def weight_parameters(model):
    return [(name, p) for name, p in model.named_parameters() if name.endswith('weight')]


def magnitude_masks(model, sparsity, scope):
    weights = weight_parameters(model)
    masks = {name: torch.ones_like(p) for name, p in weights}
    if scope == 'global':
        values = torch.cat([p.detach().abs().flatten() for _, p in weights])
        mask = torch.ones_like(values)
        count = round(sparsity*values.numel())
        mask[torch.argsort(values, stable=True)[:count]] = 0
        start = 0
        for name, p in weights:
            masks[name] = mask[start:start+p.numel()].reshape_as(p)
            start += p.numel()
    elif scope == 'layerwise':
        for name, p in weights:
            count = round(sparsity*p.numel())
            mask = masks[name].flatten()
            mask[torch.argsort(p.detach().abs().flatten(), stable=True)[:count]] = 0
    else:
        raise ValueError(scope)
    return masks


@torch.no_grad()
def apply_masks(model, masks):
    for name, p in model.named_parameters():
        if name in masks:
            p.mul_(masks[name])


@torch.no_grad()
def remove_filters(model, fraction):
    # Rank whole output filters by their L2 norms; do not merely zero them.
    c1, c2 = model.channels
    n1, n2 = max(1, round(c1*(1-fraction))), max(1, round(c2*(1-fraction)))
    keep1 = model.conv1.weight.flatten(1).norm(p=2, dim=1).argsort(descending=True)[:n1].sort().values
    keep2 = model.conv2.weight.flatten(1).norm(p=2, dim=1).argsort(descending=True)[:n2].sort().values
    compact = CNN((n1, n2), model.hidden)
    compact.conv1.weight.copy_(model.conv1.weight[keep1])
    compact.conv1.bias.copy_(model.conv1.bias[keep1])
    compact.conv2.weight.copy_(model.conv2.weight[keep2][:, keep1])
    compact.conv2.bias.copy_(model.conv2.bias[keep2])
    fc_weights = model.fc1.weight.reshape(model.hidden, c2, 7, 7)
    compact.fc1.weight.copy_(fc_weights[:, keep2].reshape(model.hidden, -1))
    compact.fc1.bias.copy_(model.fc1.bias)
    compact.fc2.load_state_dict(model.fc2.state_dict())
    # Compare with the original dimensions and matching channel-output gates.
    def gate(indices, size):
        mask = torch.zeros(1, size, 1, 1)
        mask[:, indices] = 1
        return lambda module, args, output: output*mask
    hooks = [model.conv1.register_forward_hook(gate(keep1, c1)),
             model.conv2.register_forward_hook(gate(keep2, c2))]
    sample = torch.randn(2, 1, 28, 28)
    try:
        torch.testing.assert_close(compact(sample), model(sample), rtol=1e-5, atol=1e-6)
    finally:
        for hook in hooks:
            hook.remove()
    return compact, dict(kept_conv1=keep1.tolist(), kept_conv2=keep2.tolist(),
                         actual_filter_fraction_removed=1-(n1+n2)/(c1+c2))
