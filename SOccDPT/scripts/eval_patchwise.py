import sys
import os

from tqdm import tqdm
import numpy as np
import torch
from ..model.SOccDPT import SOccDPT_versions

from .train_SOccDPT import train_net
from ..patchwise_training import PatchWiseInplace


def freeze_pretrained_encoder(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_pretrained_encoder_by_percentage(model, percentage):
    assert 0 <= percentage <= 1, "percentage must be between 0 and 1"

    parameters = list(model.parameters())
    N = len(parameters)
    M = round(N * percentage)
    unfreeze_indices = range(0, M, 1)
    for index, param in enumerate(model.parameters()):
        if index in unfreeze_indices:
            param.requires_grad = True
        else:
            param.requires_grad = False


class DummyWandB:
    def log(self, *args, **kwargs):
        pass


# Disable


def blockPrint():
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")


# Restore


def enablePrint():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


def train_Unet(
    device,
    batch_size,
    patchwise_percentage,
    encoder_percentage,
):
    # Load Unet from torchvision
    model = torch.hub.load(
        "mateuszbuda/brain-segmentation-pytorch",
        "unet",
    ).to(device)

    freeze_pretrained_encoder(model)
    unfreeze_pretrained_encoder_by_percentage(model, encoder_percentage)

    # Dummy unet input
    x = torch.randn(batch_size, 3, 256, 256).to(device)

    # Dummy unet target
    y = torch.randn(batch_size, 1, 256, 256).to(device)

    for model_patch in PatchWiseInplace(model, patchwise_percentage):

        # Dummy unet prediction
        pred = model_patch(x)

        # Dummy unet loss
        loss = torch.nn.functional.mse_loss(pred, y)

        # Dummy unet backward
        loss.backward()

    del model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "{},{},{}".format(
            "batch_size", "patchwise_percentage", "encoder_percentage"
        )
    )

    for encoder_percentage in tqdm(np.arange(0.1, 1.1, 0.1)):
        for patchwise_percentage in tqdm(
            np.arange(0.05, 1.1, 0.1), leave=False
        ):
            for batch_size in range(36, 128, 8):
                blockPrint()
                try:
                    train_Unet(
                        device=device,
                        batch_size=batch_size,
                        patchwise_percentage=patchwise_percentage,
                        encoder_percentage=encoder_percentage,
                    )
                    # Catch torch.cuda.OutOfMemoryError
                except torch.cuda.OutOfMemoryError:
                    enablePrint()
                    print(
                        "{},{},{}".format(
                            batch_size,
                            patchwise_percentage,
                            encoder_percentage,
                        )
                    )
                    break

                torch.cuda.empty_cache()
                enablePrint()


def main_soccdpt():
    experiment = DummyWandB()
    epochs = 1
    learning_rate = 0.0001
    val_percent = 0.05
    save_checkpoint = False
    amp = False
    weight_decay = 0.0
    loss_weights = [0.5, 0.5]
    dataset_percentage = 0.0025
    compute_scale_and_shift = True
    load = False
    load_depth = False
    load_seg = False
    SOccDPT_version = 3
    SOccDPT = SOccDPT_versions[SOccDPT_version]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = "dpt_swin2_tiny_256"
    checkpoint_dir = "checkpoints"
    base_path = os.path.expanduser("~/Datasets/Depth_Dataset_Bengaluru")
    dataset_name = "idd"
    project_name = "PatchWise_test"

    print(
        "{},{},{}".format(
            "batch_size", "patchwise_percentage", "encoder_percentage"
        )
    )

    for encoder_percentage in np.arange(0.1, 1.1, 0.1):
        for patchwise_percentage in np.arange(0.1, 1.1, 0.1):
            for batch_size in range(1, 16):
                blockPrint()
                try:
                    train_net(
                        experiment,
                        epochs,
                        batch_size,
                        learning_rate,
                        val_percent,
                        save_checkpoint,
                        amp,
                        weight_decay,
                        encoder_percentage,
                        patchwise_percentage,
                        loss_weights,
                        dataset_percentage,
                        compute_scale_and_shift,
                        load,
                        load_depth,
                        load_seg,
                        SOccDPT,
                        SOccDPT_version,
                        device,
                        model_type,
                        checkpoint_dir,
                        base_path,
                        dataset_name,
                        project_name,
                    )
                    # Catch torch.cuda.OutOfMemoryError
                except torch.cuda.OutOfMemoryError:
                    enablePrint()
                    print(
                        "{},{},{}".format(
                            batch_size,
                            patchwise_percentage,
                            encoder_percentage,
                        )
                    )
                    break

                torch.cuda.empty_cache()
                enablePrint()


if __name__ == "__main__":
    main()
