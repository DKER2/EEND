#!/usr/bin/env python3
from espnet2.tasks.diar_sceend import SC_EENDTask

def get_parser():
    parser = SC_EENDTask.get_parser()
    return parser


def main(cmd=None):
    r"""Diar-Enh training.
    Example:
        % python diar_enh_train.py --print_config --optim adadelta \
                > conf/train.yaml
        % python diar_enh_train.py --config conf/train.yaml
    """
    SC_EENDTask.main(cmd=cmd)


if __name__ == "__main__":
    main()