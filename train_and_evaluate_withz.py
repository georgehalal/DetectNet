import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils
import model.detnetz as net
from model.dataloader_det import DetectionDataset as DD


plt.rcParams.update({'font.size': 15, 'figure.figsize': (10, 6)})
parser = argparse.ArgumentParser()
parser.add_argument('--test_dir', default='tests/detectz1', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --test_dir containing weights to reload before training")


def train_step(model, optimizer, loss_fn, conditions, true, out):
    """One training step

    Args:
        model: the feed-forward network
        optimizer: the optimizer for the network
        loss_fn: the loss function
        conditions: the observing conditions used as inputs
        true: the true galaxy magnitudes used as inputs
        out: the ground truth output
    """

    optimizer.zero_grad()
    z = Variable(torch.randn(conditions.shape[0],1)).requires_grad_(True).cuda(non_blocking=True) #generalize by if params.cuda
    conditions.requires_grad_(True)
    true.requires_grad_(True)
    
    predout = model(conditions, true, z).squeeze()
    loss = loss_fn(predout, out)
    loss.backward()
    optimizer.step()
    return loss.item(), predout.data


def train(model, optimizer, dataloader, loss_fn, acc, params):
    """Training loop which keeps track of the metrics

    Args:
        model: the feedforward network
        optimizer: the optimizer for the network
        dataloader: the training data
        loss_fn: loss function
        acc: accuracy function
        params: hyperparameters used for training
    """

    model.train()
    
    summ = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for i, (conditions_batch, true_batch, out_batch) in enumerate(dataloader):
            
            conditions_batch, true_batch, out_batch = Variable(conditions_batch), Variable(true_batch), Variable(out_batch)
            
            if params.cuda:
                conditions_batch, true_batch, out_batch = conditions_batch.cuda(non_blocking=True), true_batch.cuda(non_blocking=True), out_batch.cuda(non_blocking=True)

            loss, predout = train_step(model, optimizer, loss_fn, conditions_batch, true_batch, out_batch)
            
            if i % params.save_summary_steps == 0:
                out_batch = out_batch.data.cpu().numpy()
                predout = (predout>=0.5).int().cpu().numpy()

                summary_batch = {'loss': loss}
                summary_batch['accuracy'] = acc(predout,out_batch)
                summ.append(summary_batch)

            loss_avg.update(loss)

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_mean


def evaluate(model, dataloader, loss_fn, acc, params):
    """Evaluate the metrics to specify when to save out the weights

    Args:
        model: the feed-forward network
        dataloader: the training data
        loss_fn: loss function
        acc: accuracy function
        params: hyperparameters used for training
    """

    model.eval()

    summ = []
    for conditions_batch, truth_batch, out_batch in dataloader:

        conditions_batch, truth_batch = Variable(conditions_batch), Variable(truth_batch)
        z_batch = Variable(torch.randn(conditions_batch.shape[0],1))

        if params.cuda:
            conditions_batch, truth_batch, out_batch, z_batch = conditions_batch.cuda(non_blocking=True), truth_batch.cuda(non_blocking=True), out_batch.cuda(non_blocking=True), z_batch.cuda(non_blocking=True)

        predout_batch = model(conditions_batch, truth_batch, z_batch).squeeze()
        loss = loss_fn(predout_batch, out_batch).item()

        out_batch = out_batch.data.cpu().numpy()
        predout = (predout_batch.data>=0.5).int().cpu().numpy()

        summary_batch = {'loss': loss}
        summary_batch['accuracy'] = acc(predout,out_batch)
        summ.append(summary_batch)

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


def make_plot(p, name, y, test_dir):
    """Plot and save metrics as a function of epochs.

    Args:
        p: metric to plot
        name: name to save plot to
        y: y-axis label name
        test_dir: the directory to save the plots to
    """

    plt.figure()
    plt.plot(p)
    plt.xlabel("Epochs")
    plt.ylabel(y)
    plt.savefig(os.path.join(test_dir, name+'.png'))


def train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, acc, params, test_dir, restore_file=None):
    """Train the model, evaluate the metrics, and save some plots.

    Args:
        model: the feed-forward network
        train_dl: the training dataset
        val_dl: the validation dataset
        optimizer: the optimizer for the model
        loss_fn: loss function
        acc: accuracy function
        params: hyperparameters used for training
        test_dir: the directory containing the testing parameters
        restore_file: file containing model parameters to load and continue training
    """

    if restore_file is not None:
        restore_path = os.path.join(test_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_acc = 0.0
    train_loss_plt = []
    val_loss_plt = []
    train_acc_plt = []
    val_acc_plt = []

    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        train_metrics = train(model, optimizer, train_dl, loss_fn, acc, params)
        val_metrics = evaluate(model, val_dl, loss_fn, acc, params)

        train_loss_plt.append(train_metrics['loss'].item())
        train_acc_plt.append(train_metrics['accuracy'].item())
        val_loss_plt.append(val_metrics['loss'].item())
        val_acc_plt.append(val_metrics['accuracy'].item())

        if epoch > 10:
            val_acc = val_metrics['accuracy']
            is_best = val_acc >= best_acc

            utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict()}, is_best=is_best, checkpoint=test_dir)

            if is_best:
                logging.info("- Found new best validation metric")
                best_acc = val_acc

                best_json_path = os.path.join(test_dir, "metrics_val_best.json")
                utils.save_dict_to_json(val_metrics, best_json_path)

        last_json_path = os.path.join(test_dir, "metrics_val_last.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

    make_plot(train_loss_plt, "train_loss", "Train Loss", test_dir)
    make_plot(train_acc_plt, "train_acc", "Train Accuracy", test_dir)
    make_plot(val_loss_plt, "val_loss", "Validation Loss", test_dir)
    make_plot(val_acc_plt, "val_acc", "Validation Accuracy", test_dir)


if __name__ == '__main__':

    args = parser.parse_args()
    json_path = os.path.join(args.test_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Check whether a GPU is available
    params.cuda = torch.cuda.is_available()

    torch.manual_seed(340)
    if params.cuda:
        torch.cuda.manual_seed(340)

    utils.set_logger(os.path.join(args.test_dir, 'train.log'))

    logging.info("Loading the datasets...")

    train_dl = DataLoader(DD("train"), batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)
    val_dl = DataLoader(DD("val"), batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)
    
    logging.info("- done.")

    # Load the network
    model = net.DetectionNet(params).cuda() if params.cuda else net.DetectionNet(params)
    
    logging.info(model)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    loss_fn = net.loss_fn
    acc = net.accuracy

    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, acc, params, args.test_dir, args.restore_file)


