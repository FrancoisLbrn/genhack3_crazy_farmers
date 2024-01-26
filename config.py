import argparse


def str2bool(v):
    return v.lower() in ('true', '1')

parser = argparse.ArgumentParser(description='argument for training')

# path and dirs
parser.add_argument('--logs_folder', type=str, default='./logs/',
                    help='folder in where logs and checkpoints are stored')
parser.add_argument('--data_path', type=str,
                    help='path to data')
parser.add_argument('--subset', type=str2bool, default=False,
                    help='whether to subset on Part 1')
# optimization
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch_size')
parser.add_argument('--num_workers', type=int, default=16,
                    help='num of workers to use')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.05,
                    help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 in Adam optimizer')
parser.add_argument('--weight_decay', type=float, default=4e-3,
                    help='weight decay (L2 penalty)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--early_stop', type=str2bool, default=True,
                    help='early stops the training if validation loss/accuracy does not improve after a given patience')
parser.add_argument('--train_patience', type=int, default=20,
                    help='number of epochs to wait before stopping training')
parser.add_argument('--flush', type=str2bool, default=False,
                    help='whether to delete ckpt + log files')
parser.add_argument('--best', type=str2bool, default=True,
                    help='load best model or most recent')
parser.add_argument('--random_seed', type=int, default=42,
                    help='general seed to ensure reproducibility')
parser.add_argument('--split_num', type=int, default=0,
                    help='split number of data')
parser.add_argument('--repeat_num', type=int, default=0,
                    help='repeat number of data')
parser.add_argument('--is_train', type=str2bool, default=True,
                    help='whether to train or test the model')
parser.add_argument('--trial_num', type=int, default=1, required=True,
                    help='trial number when recording multiple runs')
parser.add_argument('--dataset', type=str, default='data',
                    help='type of dataset')
parser.add_argument('--latent_dim', type=int, default=1, required=True,
                    help='Dimension of the latent space')

def get_config():
    config, unparsed = parser.parse_known_args()
    
    return config, unparsed

