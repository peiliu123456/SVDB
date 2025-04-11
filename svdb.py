from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit

from utils.utils import get_imagenet_r_mask

imagenet_r_mask = get_imagenet_r_mask()
import PIL
import torchvision.transforms as transforms
import my_transforms as my_transforms
import numpy as np
from torch.nn import functional as F
from vdb_loss import VDBSoftmaxLoss
import math
CE = nn.CrossEntropyLoss()


def skp(list_attentions_a, list_attentions_b, normalize=True, feature_distil_factor=None):
    assert len(list_attentions_a) == len(list_attentions_b)
    loss = torch.tensor(0.).cuda()
    for ii, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, C, embsize)
        assert a.shape == b.shape, (a.shape, b.shape)
        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        if normalize:
            a = F.normalize(a, dim=2, p=2)
            b = F.normalize(b, dim=2, p=2)
        if feature_distil_factor is None:
            layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        else:
            factor = feature_distil_factor[ii].reshape([1, -1])
            layer_loss = torch.mean(factor * torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss
    return loss / len(list_attentions_a)


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


class SVDB(nn.Module):
    def __init__(self, model, optimizer, args, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.freq = 0
        self.model0 = None
        self.model1 = None
        self.alpha = 0.95
        self.args = args
        self.importance = None
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.ema = None
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.margin_e0 = 0.4 * math.log(args.num_classes)
        self.vdb_loss = VDBSoftmaxLoss(self.model.head, num_classes=args.num_classes, scale=30.0, margin=0.4)
        self.optimizer_vdb = torch.optim.SGD(self.vdb_loss.parameters(), lr=1e-6)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, reset_flag = self.forward_and_adapt(x, self.model, self.optimizer)
            if reset_flag:
                self.reset()
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_ema,
                                 self.model_state, self.optimizer_state)
        self.old_model1 = deepcopy(self.model)
        self.old_model0 = deepcopy(self.model)
        self.model_ema = deepcopy(self.model)
        self.ema = None

    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        if self.freq % 2 == 0:
            self.old_model0 = deepcopy(model)
        else:
            self.old_model1 = deepcopy(model)

        outputs, semantic, hidden_layer = model(x)

        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            max_probs, labels = torch.max(probs, dim=1)
            mask1 = max_probs >= max_probs.mean(dim=0).item()
            entropys = entropy(outputs)
            mask2 = entropys <= entropys.mean(dim=0).item()
            mask = mask1 & mask2
            if self.args.test_batch_size == 1:
                mask3 = entropys <= self.margin_e0
                mask = mask & mask3

        labels = labels[mask]
        vdbloss = self.vdb_loss(semantic[mask], labels)
        celoss = CE(outputs[mask], labels)
        loss = vdbloss + celoss
        reset_flag = False
        if self.freq != 0:
            with torch.no_grad():
                _, __, old_hidden_layer = self.old_model1(x) if self.freq % 2 == 0 else self.old_model0(x)
            skploss = skp(hidden_layer, old_hidden_layer, feature_distil_factor=self.importance) * 10.
            loss += skploss
        self.freq += 1

        loss.backward()

        self.normalize_importance()
        self.AM_loss.margins.grad *= -1.
        self.optimizer_AM.step()
        self.optimizer_AM.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

        if not np.isnan(entropys[mask].mean().item()):
            self.ema = update_ema(self.ema,
                                  entropys[mask].mean().item())  # record moving average loss values for model recovery
        if self.ema is not None:
            em = 0.15
            if self.ema < em:
                print(f"ema < {em}, now reset the model")
                reset_flag = True
        return outputs, reset_flag

    def normalize_importance(self):
        with torch.no_grad():
            grad = []
            for i, g in enumerate(self.model.block_gradients):
                if g is not None:
                    grad.append(g)
            grad = torch.abs(torch.stack(grad))
            gradients = torch.abs(grad.mean(dim=(1, 3)))
            min_values, _ = gradients.min(dim=1, keepdim=True)
            max_values, _ = gradients.max(dim=1, keepdim=True)
            range_values = max_values - min_values
            range_values[range_values == 0] = 1e-6
            if self.importance is None:
                self.importance = (gradients - min_values) / range_values
            else:
                self.importance = ((gradients - min_values) / range_values) * self.alpha + self.importance * (
                        1 - self.alpha)
            del grad
            self.model.clear_gradients()
        return self.importance


def entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, model_ema, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=False)
    optimizer.load_state_dict(optimizer_state, strict=False)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
