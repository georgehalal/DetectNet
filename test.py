import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import utils
import model.detnet as net


plt.rcParams.update({'font.size': 15, 'figure.figsize': (10, 6)})
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/preprocessed_binary/test', \
                    help="Directory containing test dataset")
parser.add_argument('--test_dir', default='tests/detect2', \
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', \
                    help="name of the file in --test_dir containing weights to load")


def evaluate(model, true, cond, out, loss_fn, acc, test_dir):
    """Use trained model to generate ROC curve and magnitude distribution plots

    Args:
        model: the feedforward network
        true: the true galaxy magnitudes used as inputs
        cond: the observing conditions used as inputs
        out: ground truth observed galaxy magnitudes
        loss_fn: the loss function
        acc: accuracy function
        test_dir: the directory to save the plots to
    """

    model.eval()

    predout = model(cond, true).squeeze().data.cpu()
    loss = loss_fn(predout, out).item()

    out = out.numpy()

    pred = (predout>=0.5).int().numpy()
    accuracy = acc(pred, out)

    fpr, tpr, _ = roc_curve(out, predout, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    r = -2.5*np.log10(true[pred==1][:,1].numpy())+30
    i = -2.5*np.log10(true[pred==1][:,2].numpy())+30
    z = -2.5*np.log10(true[pred==1][:,3].numpy())+30

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label='AUC = {:.2f}'.format(roc_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Detection ROC Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(test_dir,'detroc.png'))

    plt.figure()
    plt.hist2d(i, r-i, bins=100, range=[[20, 25], [-2, 2]])
    plt.xlabel('$i_\mathrm{true}$')
    plt.ylabel('$(r-i)_\mathrm{true}$')
    plt.clim(0,2500)
    plt.colorbar()
    plt.savefig(os.path.join(test_dir,'ri_i_t.png'))

    plt.figure()
    plt.hist2d(i-z, r-i, bins=100, range=[[-2, 2], [-2, 2]])
    plt.xlabel('$(i-z)_\mathrm{true}$')
    plt.ylabel('$(r-i)_\mathrm{true}$')
    plt.clim(0,20000)
    plt.colorbar()
    plt.savefig(os.path.join(test_dir,'ri_iz_t.png'))

    logging.info("- Test metrics : loss = {}; accuracy = {}; \
                roc_auc = {}".format(loss, accuracy, roc_auc))
    return


if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.test_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration \
                            file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()

    utils.set_logger(os.path.join(args.test_dir, 'test.log'))

    logging.info("Loading the test dataset...")

    true = Variable(torch.load(os.path.join(args.data_dir,'true.pth')))
    cond = Variable(torch.load(os.path.join(args.data_dir,'cond.pth')))
    out = torch.load(os.path.join(args.data_dir,'out.pth'))
    
    if params.cuda:
        true, cond = true.cuda(non_blocking=True), cond.cuda(non_blocking=True)

    logging.info("- done.")

    model = net.DetectionNet(params).cuda() if params.cuda else net.DetectionNet(params)

    logging.info(model)

    loss_fn = net.loss_fn
    acc = net.accuracy

    logging.info("Starting evaluation...")

    utils.load_checkpoint(os.path.join(args.test_dir, args.restore_file + '.pth.tar'), model)

    evaluate(model, true, cond, out, loss_fn, acc, args.test_dir)

