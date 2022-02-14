# -*- coding: utf-8 -*-
"""
test.py

Test the trained model by plotting an ROC curve and the distributions
of the galaxy magnitudes.

Author: George Halal
Email: halalgeorge@gmail.com
"""


__author__ = "George Halal"
__email__ = "halalgeorge@gmail.com"


import argparse
import logging
import os
from typing import Callable

import numpy as np
import torch
from torch.autograd import Variable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import utils
import model.detnet as net


plt.rcParams.update({"font.size": 15, "figure.figsize": (10, 6)})
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/preprocessed_binary/test",
                    help="Directory containing test dataset")
parser.add_argument("--test_dir", default="tests/detect",
                    help="Directory containing params.json")
parser.add_argument("--restore_file", default="best",
                    help=("name of the file in --test_dir "
                          "containing weights to load"))


def evaluate(model: net, true: torch.tensor, cond: torch.tensor,
             out: torch.tensor,
             loss_fn: Callable[[torch.tensor, torch.tensor], torch.tensor],
             acc: Callable[[torch.tensor, torch.tensor], float],
             test_dir: str) -> None:
    """Use trained model to generate ROC curve and magnitude
    distribution plots.

    Args:
        model (net): the feedforward network
        true (torch.tensor): the true galaxy magnitudes used as inputs
        cond (torch.tensor): the observing conditions used as inputs
        out (torch.tensor): ground truth observed galaxy magnitudes
        loss_fn (Callable[[torch.tensor, torch.tensor], torch.tensor]):
            loss function
        acc (Callable[[torch.tensor, torch.tensor], float]): accuracy
            function
        test_dir (str): the directory to save the plots to.
    """

    model.eval()

    predout = model(cond, true).squeeze().data.cpu()
    loss = loss_fn(predout, out).item()

    out = out.numpy()

    pred = (predout >= 0.5).int().numpy()
    accuracy = acc(pred, out)

    fpr, tpr, _ = roc_curve(out, predout, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    r = -2.5 * np.log10(true[pred == 1][:, 1].numpy()) + 30.
    i = -2.5 * np.log10(true[pred == 1][:, 2].numpy()) + 30.
    z = -2.5 * np.log10(true[pred == 1][:, 3].numpy()) + 30.

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label="AUC = {:.2f}".format(roc_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Detection ROC Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(os.path.join(test_dir, "detroc.png"))

    plt.figure()
    plt.hist2d(i, r - i, bins=100, range=[[20, 25], [-2, 2]])
    plt.xlabel("$i_\mathrm{true}$")
    plt.ylabel("$(r-i)_\mathrm{true}$")
    plt.clim(0, 2500)
    plt.colorbar()
    plt.savefig(os.path.join(test_dir, "ri_i_t.png"))

    plt.figure()
    plt.hist2d(i - z, r - i, bins=100, range=[[-2, 2], [-2, 2]])
    plt.xlabel("$(i-z)_\mathrm{true}$")
    plt.ylabel("$(r-i)_\mathrm{true}$")
    plt.clim(0, 20000)
    plt.colorbar()
    plt.savefig(os.path.join(test_dir, "ri_iz_t.png"))

    logging.info(f"- Test metrics : loss = {loss}; accuracy = {accuracy}; "
                 f"roc_auc = {roc_auc}")

    return None


if __name__ == "__main__":
    args = parser.parse_args()
    json_path = os.path.join(args.test_dir, "params.json")
    assert os.path.isfile(json_path), ("No json configuration "
                                       f"file found at {json_path}")
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()

    utils.set_logger(os.path.join(args.test_dir, "test.log"))

    logging.info("Loading the test dataset...")

    true = Variable(torch.load(os.path.join(args.data_dir, "true.pth")))
    cond = Variable(torch.load(os.path.join(args.data_dir, "cond.pth")))
    out = torch.load(os.path.join(args.data_dir, "out.pth"))
    
    if params.cuda:
        true, cond = true.cuda(non_blocking=True), cond.cuda(non_blocking=True)

    logging.info("- done.")

    model = net.DetectionNet(params).cuda() if params.cuda else (
        net.DetectionNet(params))

    logging.info(model)

    loss_fn = net.loss_fn
    acc = net.accuracy

    logging.info("Starting evaluation...")

    utils.load_checkpoint(os.path.join(
        args.test_dir, args.restore_file + ".pth.tar"), model)

    evaluate(model, true, cond, out, loss_fn, acc, args.test_dir)

