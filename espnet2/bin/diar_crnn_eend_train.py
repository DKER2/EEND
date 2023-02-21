#!/usr/bin/env python3

from espnet2.tasks.diar_crnn_eend import CRNN_EENDTask


def get_parser():
    parser = CRNN_EENDTask.get_parser()
    return parser


def main(cmd=None):
    r"""Speaker diarization training.

    Example:
        % python diar_train.py diar --print_config --optim adadelta \
                > conf/train_diar.yaml
        % python diar_train.py --config conf/train_diar.yaml
    """
    CRNN_EENDTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
