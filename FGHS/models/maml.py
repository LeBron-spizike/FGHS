
import torch
import traceback
from learn_2learn.base_learner import BaseLearner
from learn_2learn.init import clone_module, update_module
from torch.autograd import grad


def maml_update(model, lr, grads=None, adaptable_weights=None, anil=False):
    if adaptable_weights is None:
        adaptable_weights = [p for p in model.parameters()]

    if grads is not None:
        params = adaptable_weights
        if not len(grads) == len(list(params)):
            msg = "WARNING:maml_update(): Parameters and gradients have different length. ("
            msg += str(len(params)) + " vs " + str(len(grads)) + ")"
            print(msg)
        for i, (p, g) in enumerate(zip(params, grads)):
            if g is not None:
                if anil and i < (len(params) - 2):
                    g = torch.zeros_like(g)
                p.update = - lr * g
    return update_module(model)


class MAML(BaseLearner):

    def __init__(
            self,
            model,
            lr,
            first_order=False,
            allow_unused=None,
            allow_nograd=False,
            anil=False,
    ):
        super(MAML, self).__init__()
        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused
        self.anil = anil

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(
            self, loss, adaptable_weights=None, first_order=None, allow_unused=None, allow_nograd=None, anil=None
    ):
        if anil is None:
            anil = self.anil
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        if adaptable_weights is None:
            adaptable_weights = [p for p in self.module.parameters()]

        if allow_nograd:
            # Compute relevant gradients
            diff_params = [p for p in adaptable_weights if p.requires_grad]
            grad_params = grad(
                loss,
                diff_params,
                retain_graph=second_order,
                create_graph=second_order,
                allow_unused=allow_unused,
            )
            gradients = []
            grad_counter = 0

            # Handles gradients for non-differentiable parameters
            for param in self.module.parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(
                    loss,
                    adaptable_weights,
                    retain_graph=second_order,
                    create_graph=second_order,
                    allow_unused=allow_unused,
                )
            except RuntimeError:
                traceback.print_exc()
                print(
                    "learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?"
                )

        # Update the module
        self.module = maml_update(self.module, self.lr, gradients, adaptable_weights, anil)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None, anil=None):
    
        if anil is None:
            anil = self.anil
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAML(
            clone_module(self.module),
            lr=self.lr,
            first_order=first_order,
            allow_unused=allow_unused,
            allow_nograd=allow_nograd,
            anil=anil
        )
