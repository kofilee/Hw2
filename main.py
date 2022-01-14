from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from model import classifier
from data_loader import data_loader


def parse_args():
    desc = "Hw2 by Kofi"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-type',
        '--net_type',
        help='Net type (DNN or CNN or RNN) (default = RNN)',
        default='RNN',
        type=str,
        metavar='')
    parser.add_argument(
        '-t',
        '--timesteps',
        help='Number of time steps (default = 20)',
        default=20,
        type=int,
        metavar='')
    parser.add_argument(
        '-b',
        '--batch_size',
        help='Batch size (default = 16)',
        default=16,
        type=int,
        metavar='')
    parser.add_argument(
        '-e',
        '--epoch',
        help='number of epoch (default = 50)',
        default=50,
        type=int,
        metavar='')
    parser.add_argument(
        '-te',
        '--te',
        help='Testing (default = False)',
        default=0,
        type=int,
        metavar='')
    parser.add_argument(
        '-F',
        '--Fault',
        help='Fault type (default 0 ~ 14)',
        default=[0, 2, 12, 14],
        nargs='+',
        type=int,
        metavar='')

    return check_args(parser.parse_args())


def check_args(args):
    try:
        assert args.epoch >= 1
    except ValueError:
        print('number of epochs must be larger than or equal to one')
        raise

    try:
        assert any(x for x in args.Fault if 0 <= x <= 20)
    except ValueError:
        print('batch size must be larger than or equal to one')
        raise

    return args


def main():
    args = parse_args()

    if args is None:
        exit()

    timesteps = args.timesteps
    batch_size = args.batch_size
    epoch = args.epoch
    net_type = args.net_type

    train_data, train_label = data_loader(timesteps, [0, 2, 12, 14])
    test_data, test_label = data_loader(timesteps, [0, 2, 12, 14], 1)

    rnd_index = np.arange(train_data.shape[0])
    np.random.shuffle(rnd_index)
    train_data = train_data[rnd_index]
    train_label = train_label[rnd_index]

    classification_model = classifier(net_type)
    classification_model.model.compile(
        optimizer=keras.optimizers.Adam(10e-5),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()]
    )
    history = classification_model.model.fit(train_data, train_label,
                                             batch_size=batch_size,
                                             epochs=epoch,
                                             validation_split=0.2,
                                             )

    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    results = classification_model.model.evaluate(test_data, test_label, batch_size=128)

    print(f"test loss: {results[0]}, test acc: {results[1]}")

    return 0


if __name__ == '__main__':
    main()
