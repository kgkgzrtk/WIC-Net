"""
Weather image creation.

Usage:
    main.py [--model_name=MODEL_N] [--epoch=EPOCH] [--lr=LR] [--dataset_dir=DATASET_DIR] [--batch_size=BATCH_SIZE] [--dim=DIM] [--tensorboard_dir=TB_DIR] [--checkpoint_dir=CP_DIR]
    main.py -h | --help

Option:
    -h, --help      : Show this screen.
    --model_name=MODEL_N   : The name of model. [default: wic-net]
    --epoch=EPOCH   : The number of epoch to train on. [default: 1000]
    --lr=LR         : Learning rate. [default: 2e-4]
    --dataset_dir=DATASET_DIR   : The path of the dataset directory. [default: ./data]
    --tensorboard_dir = TB_DIR  : The path of TensorBoard directory. [default: ./results/tensorboard]
    --checkpoint_dir = CP_DIR  : The path of checkpoint directory. [default: ./results/checkpoint]
    --batch_size=BATCH_SIZE     : The number of batch size. [default: 16]
    --dim=DIM       : The channel dimention of model. [default: 16]

"""
import tensorflow as tf
from docopt import docopt
from model import *

def main():
    args = docopt(__doc__)
    print(args)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list=""
    #config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        model = wic_model(sess, args)
        model.train()
        sess.close()

if __name__ == "__main__":
    main()
