import os

import torch

from ..model.loader import load_model
from ..model.SOccDPT import SOccDPT_versions, model_types


@torch.no_grad()
def main(args):
    SOccDPT = SOccDPT_versions[args.version]
    device = torch.device(args.device)

    net = load_model(
        arch=SOccDPT,
        model_kwargs=dict(path=args.load),
        device=device,
        optimize=args.optimize,
    )

    if args.compile:
        net = torch.compile(net)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SOccDPT")
    parser.add_argument(
        "-v",
        "--version",
        choices=[1, 2, 3],
        required=True,
        help="SOccDPT version",
    )

    parser.add_argument(
        "-dt",
        "--dataset",
        choices=["bdd", "idd"],
        required=True,
        help="Dataset to train using",
    )

    parser.add_argument(
        "-t",
        "--model_type",
        choices=model_types,
        required=True,
        help="Model architecture to use",
    )

    parser.add_argument(
        "-d",
        "--device",
        # default="cuda:0" if torch.cuda.is_available() else "cpu",
        default="cpu",
        help="Device to use for training",
    )

    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        default=os.path.join(os.getcwd(), "checkpoints"),
        help="Directory to save checkpoints in",
    )

    parser.add_argument(
        "-cm",
        "--compile",
        action="store_true",
        help="Use torch.compile to optimize the model",
    )

    parser.add_argument(
        "-o",
        "--optimize",
        action="store_true",
        help="Optimize the model using net.half",
    )

    parser.add_argument(
        "-b",
        "--base_path",
        default=os.path.expanduser("~/Datasets/Depth_Dataset_Bengaluru"),
        help="Base path to dataset",
    )

    args = parser.parse_args()

    main(args=args)
