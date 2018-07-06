"""
Weather image creation.

Usage:
    main.py [--epoch=EPOCH] [--lr=LR] [--dataset_dir=DATASET_DIR] [--batch_size=BATCH_SIZE] [--dim=DIM] [--tensorboard_dir=TB_DIR]
    main.py -h | --help

Option:
    -h, --help      : Show this screen.
    --epoch=EPOCH   : The number of epoch to train on. [default: 1000]
    --lr=LR         : Learning rate. [default: 2e-4]
    --dataset_dir=DATASET_DIR   : The path of the dataset directory. [default: ./data]
    --tensorboard_dir = TB_DIR  : The path of TensorBoard directory. [default: ./results/tensorboard]
    --batch_size=BATCH_SIZE     : The number of batch size. [default: 16]
    --dim=DIM       : The channel dimention of model. [default: 16]

"""
import tensorflow as tf
from docopt import docopt
from model import *

def main():
    args = docopt(__doc__)
    print(args)
    with tf.Session() as sess:
        model = wic_model(sess, args)
        model.train()
        sess.close()

if __name__ == "__main__":
    main()
