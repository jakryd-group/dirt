# -*- coding: utf-8 -*-
""" Argument parser for user-friendly command-line interfaces """

from argparse import ArgumentParser
from argparse import RawTextHelpFormatter


def boolean_string(s):
    """
    this function is needed to parse False values from CLI
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def parser(desc='dirt: MNIST denoising using AE',
           n_epochs=25,
           batch_size=100,
           batch_test_size=100,
           noise=0.2,
           normalize=True,
           learn=0.001,
           weight_decay=5e-4,
           verbose=True,
           tensorboard=True,
           suppress=False):
    """
    This function create parser object and prepares it with default values.
    None default values can be passed as function parameters.
    """

    pars = ArgumentParser(
            usage='python %(prog)s [options]',
            description=desc,
            formatter_class=RawTextHelpFormatter)

    pars.add_argument('--n_epochs', type=int, required=False, default=n_epochs,
                help='number of epochs \t\t\t def: %s' % n_epochs, metavar='')

    pars.add_argument('--batch_size', type=int, required=False, default=batch_size,
                help='batch size \t\t\t def: %s' % batch_size, metavar='')

    pars.add_argument('--batch_size_test', type=int, required=False, default=batch_test_size,
                help='batch size for testing \t\t def: %s' % batch_test_size, metavar='')

    pars.add_argument('--noise', type=float, required=False, default=noise,
                help='noise level \t\t\t def: %s' % noise, metavar='')

    pars.add_argument('--normalize_img', default=normalize, type=boolean_string,
                help='normalize image with noise \t def: %s' % normalize, metavar='')

    pars.add_argument('--learning_rate', type=float, required=False,
                default=learn, help='learning rate \t\t\t def: %s' % learn, metavar='')

    pars.add_argument('--weight_decay', type=float, required=False,
                default=weight_decay, help='weight decay \t\t\t def: %s' % weight_decay, metavar='')

    pars.add_argument('--verbose', default=verbose, type=boolean_string,
                help='verbose mode \t\t\t def: %s' % verbose, metavar='')

    pars.add_argument('--tensorboard', default=tensorboard, type=boolean_string,
                help='store training progress \t\t def: %s' % tensorboard, metavar='')

    pars.add_argument('--suppress', default=suppress, type=boolean_string,
                help='suppress all info \t\t def: %s' % suppress, metavar='')

    return pars.parse_args()
