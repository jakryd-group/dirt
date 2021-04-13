from argparse import ArgumentParser

parser = ArgumentParser(usage='python %(prog)s [options]',
            description='dirt: MNIST denoising using AE')
parser.add_argument('--n_epochs', type=int, required=False, default=10,
            help='number of epochs')
parser.add_argument('--batch_size', type=int, required=False, default=128,
            help='batch size')
parser.add_argument('--batch_size_test', type=int, required=False, default=10,
            help='batch size for testing')
parser.add_argument('--noise', type=float, required=False, default=0.2,
            help='noise level')
parser.add_argument('--normalize_img', type=bool, required=False, default=True,
            help='normalize image with noise')
parser.add_argument('--learning_rate', type=float, required=False,
            default=0.001, help='learning rate')
parser.add_argument('--verbose', type=bool, required=False, default=True,
            help='verbose mode')

args = parser.parse_args()