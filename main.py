"""
Weather image creation.

Usage:
    main.py [--model_name=MODEL_N] [--epoch=EPOCH] [--lr=LR] [--dataset_dir=DATASET_DIR] [--batch_size=BATCH_SIZE] [--dim=DIM] [--tensorboard_dir=TB_DIR] [--checkpoint_dir=CP_DIR] [--load_name=FILE_NAME] [--max_lmda=LAMBDA] [--reg_scale=REG_S] [--gpu=GPU_ID]
    main.py -h | --help

Option:
    -h, --help      : Show this screen.
    --model_name=MODEL_N   : The name of model. [default: wic-net]
    --load_name=FILE_NAME :The name of file for restore learned model.
    --epoch=EPOCH: The number of iteration to train on. [default: 128]
    --lr=LR         : Learning rate. [default: 1e-4]
    --dataset_dir=DATASET_DIR   : The path of the dataset directory. [default: ./data]
    --tensorboard_dir = TB_DIR  : The path of TensorBoard directory. [default: ./results/tensorboard]
    --checkpoint_dir = CP_DIR  : The path of checkpoint directory. [default: ./results/checkpoint]
    --batch_size=BATCH_SIZE     : The number of batch size. [default: 16]
    --dim=DIM       : The channel dimention of model. [default: 32]
    --max_lmda=LAMBDA   :the learning param. [default: 1.]
    --reg_scale=REG_S   :the learning param. [default: 0]
    --gpu=GPU_ID   : The ID of GPU for using [default: 1]

"""
import tensorflow as tf
from docopt import docopt
from model import *

def main():
    args = docopt(__doc__)
    print(args)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list=args['--gpu']
    #config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        model = wic_model(sess, args)
        model.build()
        if model.load_name is None:
            model.load_data()
            model.write_tfr('data.tfrecord')
            model.train()
        else:
            model.load_model()
            model.load_data()
            #model.weather_run(model.test_image[:model.batch_size], model.test_attr[:model.batch_size], 2, save=True)
            model.train()
        sess.close()

if __name__ == "__main__":
    main()
