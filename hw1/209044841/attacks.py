import torch
import torch.nn as nn
import torch.nn.functional as F


class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """

    def __init__(
        self,
        model,
        eps=8 / 255.0,
        n=50,
        alpha=1 / 255.0,
        rand_init=True,
        early_stop=True,
    ):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels
        in case of untargeted attacks, and the target labels in case of targeted
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally
        performs random initialization and early stopping, depending on the
        self.rand_init and self.early_stop flags.
        """

        adv = x
        adv.requires_grad = True

        if self.rand_init:
            adv = adv + torch.empty_like(adv).uniform_(-self.eps, self.eps)
            adv = torch.clamp(adv, min=0, max=1)

        for _ in range(self.n):
            outputs = self.model(adv)

            if self.early_stop:
                _, predicted = torch.max(outputs.data, 1)
                if (targeted and (predicted == y).all()) or (
                    not targeted and (predicted != y).all()
                ):
                    break

            loss = (-1 if targeted else 1) * self.loss_func(outputs, y)
            grad = torch.autograd.grad(loss.sum(), adv)[0]

            delta = (adv - x) + self.alpha * grad.sign()
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)

            adv = x + delta
            adv = torch.clamp(adv, min=0, max=1)

        return adv


class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss,
    where gradients are estimated using Natural Evolutionary Strategies
    (NES).
    """

    def __init__(
        self,
        model,
        eps=8 / 255.0,
        n=50,
        alpha=1 / 255.0,
        momentum=0.0,
        k=200,
        sigma=1 / 255.0,
        rand_init=True,
        early_stop=True,
    ):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
             historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma = sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

    def nes_est_grad(self, x, y):
        g = 0

        for _ in range(self.k):
            u = torch.randn_like(x)

            x1 = torch.clamp(x + self.sigma * u, min=0, max=1)
            x2 = torch.clamp(x - self.sigma * u, min=0, max=1)

            with torch.no_grad():
                l1 = self.loss_func(self.model(x1), y)
                l2 = self.loss_func(self.model(x2), y)

            g += l1.view(-1, 1, 1, 1) * u
            g -= l2.view(-1, 1, 1, 1) * u

        return g / (2 * self.k * self.sigma)

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels
        in case of untargeted attacks, and the target labels in case of targeted
        attacks. The method returns:
        1- The adversarially perturbed samples, which lie in the ranges [0, 1]
            and [x-eps, x+eps].
        2- A vector with dimensionality len(x) containing the number of queries for
            each sample in x.
        """

        adv = x
        adv.requires_grad = True

        if self.rand_init:
            adv = adv + torch.empty_like(adv).uniform_(-self.eps, self.eps)
            adv = torch.clamp(adv, min=0, max=1)

        grad = torch.zeros_like(adv)
        cont = torch.ones(len(adv), dtype=torch.bool)
        qs = torch.zeros(len(adv), dtype=torch.int)

        for _ in range(self.n):
            # estimate gradient, and incorporate using the momentum
            est_grad = self.nes_est_grad(adv[cont], y[cont])
            grad[cont] = self.momentum * grad[cont] + (1 - self.momentum) * est_grad
            qs[cont] += 2 * self.k

            # pgd step and projection
            delta = (adv[cont] - x[cont]) + (-1 if targeted else 1) * self.alpha * grad[
                cont
            ].sign()
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)
            adv[cont] = x[cont] + delta
            adv[cont] = torch.clamp(adv[cont], min=0, max=1)

            if self.early_stop:
                outputs = self.model(adv)
                _, predicted = torch.max(outputs.data, 1)

                if targeted:
                    # a nice trick to do && operator between tensors
                    cont = cont == (predicted != y).cpu()
                else:
                    cont = cont == (predicted == y).cpu()

                if not cont.any():
                    break

        return adv, qs


class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the
    cross-entropy loss
    """

    def __init__(
        self,
        models,
        eps=8 / 255.0,
        n=50,
        alpha=1 / 255.0,
        rand_init=True,
        early_stop=True,
    ):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss()

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels
        in case of untargeted attacks, and the target labels in case of targeted
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps].
        """

        adv = x
        adv.requires_grad = True

        if self.rand_init:
            adv = adv + torch.empty_like(adv).uniform_(-self.eps, self.eps)
            adv = torch.clamp(adv, min=0, max=1)

        for _ in range(self.n):
            grad = torch.zeros_like(adv)

            completed_models = 0

            for model in self.models:
                outputs = model(adv)

                if self.early_stop:
                    _, predicted = torch.max(outputs.data, 1)
                    if (targeted and (predicted == y).all()) or (
                        not targeted and (predicted != y).all()
                    ):
                        completed_models += 1

                loss = (-1 if targeted else 1) * self.loss_func(outputs, y)
                grad += torch.autograd.grad(loss.sum(), adv)[0]

            if completed_models == len(self.models):
                break

            delta = (adv - x) + self.alpha * grad.sign()
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)

            adv = x + delta
            adv = torch.clamp(adv, min=0, max=1)

        return adv
