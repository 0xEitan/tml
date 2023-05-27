import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x


def free_adv_train(
    model,
    data_tr,
    criterion,
    optimizer,
    lr_scheduler,
    eps,
    device,
    m=4,
    epochs=100,
    batch_size=128,
    dl_nw=10,
):
    """
    Free adversarial training, per Shafahi et al.'s work.
    Arguments:
    - model: randomly initialized model
    - data_tr: training dataset
    - criterion: loss function (e.g., nn.CrossEntropyLoss())
    - optimizer: optimizer to be used (e.g., SGD)
    - lr_scheduer: scheduler for updating the learning rate
    - eps: l_inf epsilon to defend against
    - device: device used for training
    - m: # times a batch is repeated
    - epochs: "virtual" number of epochs (equivalent to the number of
        epochs of standard training)
    - batch_size: training batch_size
    - dl_nw: number of workers in data loader
    Returns:
    - trained model
    """
    # init data loader
    loader_tr = DataLoader(
        data_tr, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=dl_nw
    )

    # init delta (adv. perturbation)
    delta = torch.zeros(batch_size, requires_grad=True).to(device)

    # total number of updates
    total = 0

    # when to update lr
    scheduler_step_iters = int(np.ceil(len(data_tr) / batch_size))

    # train
    for epoch in tqdm(range(int(np.ceil(epochs / m)))):
        for batch_i, minibatch in enumerate(loader_tr, 0):
            inputs, labels = minibatch[0].to(device), minibatch[1].to(device)
            size = inputs.size(0)

            for m_i in range(m):
                # noise = Variable(delta[:inputs.size(0)], requires_grad=True).to(device)
                # noisy_input = inputs + noise

                # perturb
                inputs = inputs + delta[:size]

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # update perturbation
                delta[:size] += eps * torch.sign(delta.grad)
                delta = torch.clamp(delta, -eps, eps)

                # noise_update = eps*torch.sign(noise.grad)
                # delta[:inputs.size(0)] += noise_update
                # delta.clamp_(-eps, eps)

                # optimize
                optimizer.step()

        # update scheduler
        if epoch > 0 and epoch % scheduler_step_iters == 0:
            lr_scheduler.step()

    # done
    return model


class SmoothedModel:
    """
    Use randomized smoothing to find L2 radius around sample x,
    s.t. the classification of x doesn't change within the L2 ball
    around x with probability >= 1-alpha.
    """

    ABSTAIN = -1

    def __init__(self, model, sigma):
        self.model = model
        self.sigma = sigma

    def _sample_under_noise(self, x, n, batch_size):
        """
        Classify input x under noise n times (with batch size
        equal to batch_size) and return class counts (i.e., an
        array counting how many times each class was assigned the
        max confidence).
        """
        # FILL ME
        pass

    def certify(self, x, n0, n, alpha, batch_size):
        """
        Arguments:
        - model: pytorch classification model (preferably, trained with
            Gaussian noise)
        - sigma: Gaussian noise's sigma, for randomized smoothing
        - x: (single) input sample to certify
        - n0: number of samples to find prediction
        - n: number of samples for radius certification
        - alpha: confidence level
        - batch_size: batch size to use for inference
        Outputs:
        - prediction / top class (ABSTAIN in case of abstaining)
        - certified radius (0. in case of abstaining)
        """

        # find prediction (top class c) - FILL ME

        # compute lower bound on p_c - FILL ME

        # done
        return c, radius


class NeuralCleanse:
    """
    A method for detecting and reverse-engineering backdoors.
    """

    def __init__(
        self, model, dim=(1, 3, 32, 32), lambda_c=0.0005, step_size=0.005, niters=2000
    ):
        """
        Arguments:
        - model: model to test
        - dim: dimensionality of inputs, masks, and triggers
        - lambda_c: constant for balancing Neural Cleanse's objectives
            (l_class + lambda_c*mask_norm)
        - step_size: step size for SGD
        - niters: number of SGD iterations to find the mask and trigger
        """
        self.model = model
        self.dim = dim
        self.lambda_c = lambda_c
        self.niters = niters
        self.step_size = step_size
        self.loss_func = nn.CrossEntropyLoss()

    def find_candidate_backdoor(self, c_t, data_loader, device):
        """
        A method for finding a (potential) backdoor targeting class c_t.
        Arguments:
        - c_t: target class
        - data_loader: DataLoader for test data
        - device: device to run computation
        Outputs:
        - mask:
        - trigger:
        """
        # randomly initialize mask and trigger in [0,1] - FILL ME

        # run self.niters of SGD to find (potential) trigger and mask - FILL ME

        # done
        return mask, trigger
