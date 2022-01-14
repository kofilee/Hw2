from __future__ import absolute_import, division, print_function, unicode_literals
import argparse

def parse_args():
    desc = "Tensorflow by Kofi"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-t',
        '--timesteps',
        help='number of time steps (default = 20)',
        default=20,
        type=int,
        metavar='')
    parser.add_argument(
        '-e',
        '--epoch',
        help='number of epoch (default = 500)',
        default=500,
        type=int,
        metavar='')
    parser.add_argument(
        '-r',
        '--restore',
        help='To restore model from cpk (default = False)',
        default=1,
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
        assert args.runs >= 2
    except ValueError:
        print('runs must be larger than or equal to two')
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

    parameters = {'timesteps': args.timesteps,
                  'epoch': args.epoch,
                  'restore': args.restore,
                  'te': args.te,
                  'Fault': args.Fault,
                  }



    return 0


if __name__ == '__main__':
    main()
