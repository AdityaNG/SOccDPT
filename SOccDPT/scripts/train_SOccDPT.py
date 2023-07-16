import json
import os
import random
import traceback
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import optim
from torch.utils.data import random_split
from tqdm import tqdm

from ..datasets.bengaluru_driving_dataset import (
    OCTraN_Depth_Segmentation,
    class_2_color,
    get_bdd_dataset,
)
from ..loss import (
    freeze_pretrained_encoder,
    unfreeze_pretrained_encoder_by_percentage,
)
from ..loss.ssi_loss import ScaleAndShiftInvariantLoss
from ..model.loader import load_model, load_transforms
from ..model.SOccDPT import DepthNet, SegNet, SOccDPT_versions, model_types
from ..patchwise_training import PatchWiseInplace
from ..utils import evaluate, get_batch


def train_net_wandb():
    experiment = wandb.init(resume="allow", anonymous="must")

    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    val_percent = wandb.config.val_percent
    save_checkpoint = wandb.config.save_checkpoint
    amp = wandb.config.amp
    weight_decay = wandb.config.weight_decay
    encoder_percentage = wandb.config.encoder_percentage
    patchwise_percentage = wandb.config.patchwise_percentage
    loss_weights = wandb.config.loss_weights
    dataset_percentage = wandb.config.dataset_percentage
    load_depth = wandb.config.load_depth
    load = wandb.config.load

    version = wandb.config.version
    device = wandb.config.device
    model_type = wandb.config.model_type
    device = torch.device(device)
    checkpoint_dir = wandb.config.checkpoint_dir
    project_name = wandb.config.project_name
    base_path = wandb.config.base_path

    SOccDPT = SOccDPT_versions[version]

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
            load_depth=load_depth,
            load=load,
            SOccDPT=SOccDPT,
            device=device,
            model_type=model_type,
            checkpoint_dir=checkpoint_dir,
            base_path=base_path,
            project_name=project_name,
        )
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        raise ex


def train_net(
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
    load,
    SOccDPT,
    device,
    model_type,
    checkpoint_dir,
    base_path,
    project_name,
):
    dir_checkpoint = os.path.join(checkpoint_dir, project_name)
    device_cpu = torch.device("cpu")

    # make sure all loss_weights are float
    assert type(loss_weights) == list, "loss_weights must be a list"
    loss_weights = [float(loss_w) for loss_w in loss_weights]
    for loss_w in loss_weights:
        assert loss_w >= 0.0, "loss_weights must be >= 0.0"

    loss_depth_w, loss_seg_w = loss_weights

    # Clear memory
    torch.cuda.empty_cache()

    # REPRODUCIBILITY
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    wandb_run_id = wandb.run.id
    print("wandb_run_id", wandb_run_id)

    # (Initialize logging)
    wandb.config.update(
        dict(
            amp=amp,
            epochs=epochs,
            batch_size=batch_size,
            val_percent=val_percent,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            save_checkpoint=save_checkpoint,
        )
    )

    # Load net
    transforms, net_w, net_h = load_transforms(
        model_type=model_type,
    )
    net = load_model(
        arch=SOccDPT,
        model_kwargs=dict(path=load),
        device=device_cpu,
    )

    net = net.to(device=device)
    net = torch.compile(net)

    # Clear memory
    torch.cuda.empty_cache()

    print("net", type(net))
    freeze_pretrained_encoder(net)
    unfreeze_pretrained_encoder_by_percentage(net, encoder_percentage)
    # freeze_pretrained_encoder(net.depth_net)
    # unfreeze_pretrained_encoder_by_percentage(
    #   net.depth_net, encoder_percentage
    # )

    print("net all params")
    mem_params = sum(
        [param.nelement() * param.element_size() for param in net.parameters()]
    )
    mem_bufs = sum(
        [buf.nelement() * buf.element_size() for buf in net.buffers()]
    )
    mem = mem_params + mem_bufs  # in bytes
    print("mem", mem / 1024.0 / 1024.0, " MB")

    print("net trainable params")
    mem_params = sum(
        [
            param.nelement() * param.element_size()
            for param in net.parameters()
            if param.requires_grad
        ]
    )
    mem_bufs = sum(
        [
            buf.nelement() * buf.element_size()
            for buf in net.buffers()
            if buf.requires_grad
        ]
    )
    mem = mem_params + mem_bufs  # in bytes
    print("mem", mem / 1024.0 / 1024.0, " MB")

    # 1. Create dataset
    dataset = get_bdd_dataset(OCTraN_Depth_Segmentation, transforms, base_path)
    total_size = len(dataset)
    total_use = int(round(total_size * dataset_percentage))
    total_discard = total_size - total_use
    dataset, _ = random_split(
        dataset,
        [total_use, total_discard],
        generator=torch.Generator().manual_seed(0),
    )
    print("len(dataset)", len(dataset))

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    assert n_val > 0, "Validation set is 0"
    assert n_train > 0, "Train set is 0"

    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    assert len(val_set) > 0, "Validation set is 0"
    assert len(train_set) > 0, "Train set is 0"

    print(
        f"""Starting training:
        epochs: {wandb.config.epochs}
        batch_size: {wandb.config.batch_size}
        learning_rate: {wandb.config.learning_rate}
        val_percent: {wandb.config.val_percent}
        save_checkpoint: {wandb.config.save_checkpoint}
        amp: {wandb.config.amp}
    """
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler
    # and the loss scaling for AMP
    optimizer = optim.Adam(
        net.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay,
        amsgrad=False,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=2
    )  # goal: minimize the loss
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    SSILoss_criterion = ScaleAndShiftInvariantLoss()
    BCE_criterion = torch.nn.BCELoss(reduction="mean")

    global_step = 0

    def criterion_disp(y_pred, y, mask):
        return SSILoss_criterion(y_pred, y, mask)

    def criterion_seg(y_pred, y, mask):
        # Apply loss mask
        masked_y_pred = torch.masked_select(y_pred, mask)
        masked_y = torch.masked_select(y, mask)
        return BCE_criterion(masked_y_pred, masked_y)

    disp_wrapper = DepthNet(net)
    seg_wrapper = SegNet(net)

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(
            total=len(train_set), desc=f"Epoch {epoch}/{epochs}", unit="img"
        ) as pbar:
            for batch_index in range(batch_size, len(train_set), batch_size):
                try:
                    torch.cuda.empty_cache()
                    batch = get_batch(train_set, batch_index, batch_size)

                    x, x_raw, mask_disp, y_disp, mask_seg, y_seg = batch
                    x = x.to(device=device, dtype=torch.float32)
                    y_disp = y_disp.to(device=device, dtype=torch.float32)
                    y_seg = y_seg.to(device=device, dtype=torch.float32)
                    mask_disp = mask_disp.to(device=device, dtype=torch.bool)
                    mask_seg = mask_seg.to(device=device, dtype=torch.bool)

                    for net_patch in PatchWiseInplace(
                        net, patchwise_percentage
                    ):
                        with torch.cuda.amp.autocast(enabled=amp):
                            y_disp_pred, y_seg_pred, points = net_patch(x)

                            if len(y_seg_pred.shape) == 3:
                                y_seg_pred = y_seg_pred.unsqueeze(0)

                            if len(y_disp_pred.shape) == 2:
                                y_disp_pred = y_disp_pred.unsqueeze(0)

                            loss_disp = criterion_disp(
                                y_disp_pred, y_disp, mask_disp
                            )
                            loss_seg = criterion_seg(
                                y_seg_pred, y_seg, mask_seg
                            )
                            loss = (
                                loss_depth_w * loss_disp
                                + loss_seg_w * loss_seg
                            )

                        optimizer.zero_grad(set_to_none=True)
                        grad_scaler.scale(loss).backward()
                        grad_scaler.step(optimizer)
                        grad_scaler.update()

                    pbar.update(batch_size)
                    epoch_loss += loss.item()
                    experiment.log(
                        {
                            "train_loss": loss.item(),
                            "step": global_step,
                            "epoch": epoch,
                        }
                    )
                    pbar.set_postfix(**{"loss (batch)": loss.item()})

                    # Evaluation round
                    division_step = n_train // (3 * batch_size)
                    # if division_step >= 0:
                    if global_step % division_step == 0:
                        evaluate(
                            net,
                            seg_wrapper,
                            disp_wrapper,
                            val_set,
                            device,
                            amp,
                            x_raw,
                            y_disp,
                            y_disp_pred,
                            y_seg,
                            y_seg_pred,
                            points,
                            class_2_color,
                            loss,
                            optimizer.param_groups[0]["lr"],
                            global_step,
                            epoch,
                            experiment,
                        )
                        scheduler.step(loss)
                    global_step += 1
                except Exception as ex:
                    print(ex)
                    traceback.print_exc()
                    raise ex

            if save_checkpoint:
                dir_checkpoint_run = os.path.join(dir_checkpoint, wandb_run_id)
                Path(dir_checkpoint_run).mkdir(parents=True, exist_ok=True)
                torch.save(
                    net.state_dict(),
                    str(
                        os.path.join(
                            dir_checkpoint_run,
                            "checkpoint_epoch_{}.pth".format(epoch),
                        )
                    ),
                )
                print(f"Checkpoint {epoch} saved!")


def main(args):
    with open(args.sweep_json, "r") as sweep_json_file:
        sweep_config = json.load(sweep_json_file)
        sweep_config["device"] = args.device
        sweep_config["version"] = args.version
        sweep_config["model_type"] = args.model_type
        sweep_config["checkpoint_dir"] = args.checkpoint_dir

    project_name = "SOccDPT_V{version}_{model_type}".format(
        version=str(args.version), model_type=args.model_type
    )

    sweep_config["project_name"] = project_name

    sweep_id = wandb.sweep(
        sweep_config, project=project_name, entity="pw22-sbn-01"
    )
    print("sweep_id", sweep_id)
    wandb.agent(sweep_id, function=train_net_wandb, count=1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SOccDPT")
    parser.add_argument(
        "-v", "--version", choices=[1, 2, 3], help="SOccDPT version"
    )

    parser.add_argument(
        "-t",
        "--model_type",
        choices=model_types,
        help="Model architecture to use",
    )

    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )

    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        default=os.path.join(os.getcwd(), "checkpoints"),
        help="Directory to save checkpoints in",
    )

    parser.add_argument(
        "-b",
        "--base_path",
        default=os.path.expanduser("~/Datasets/Depth_Dataset_Bengaluru"),
        help="Base path to dataset",
    )

    parser.add_argument(
        "--sweep_json", help="Path to checkpoint to sweep json"
    )

    args = parser.parse_args()

    main(args=args)
