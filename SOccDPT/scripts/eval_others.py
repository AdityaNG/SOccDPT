import os

import torch
from torch.utils.data import random_split

from ..model.SOccDPT import DepthNet , SegNet , SOccDPT
from ..utils import (
    evaluate_depth,
    evaluate_seg
)

from ..datasets.bengaluru_driving_dataset import (
    BDD_Depth_Segmentation,
    class_2_color as class_2_color_bdd,
    get_bdd_dataset,
)
from ..datasets.idd import (
    get_all_IDD_Depth_Segmentation_datasets,
)
from ..datasets.anue_labels import (
    # LEVEL1_ID,
    # level1_to_class,
    # level1_to_color as class_2_color_idd,

    LEVEL4_BASICS_ID,
    level4_basics_to_class,
    level4_basics_to_color as class_2_color_idd,
)

from ..utils import (
    blockPrint, enablePrint
)

import numpy as np
import random
import cv2
from tqdm import tqdm
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

other_models = [
    'DPT_SwinV2_T_256',     # midas
    'DPT_Hybrid',           # midas
    'DPT_Large',            # midas
    'monodepth',
    'monodepth2',
    'manydepth',
    'zerodepth',
    'packnet',
]


class OtherModelWrapper(SOccDPT):
    """
    Unifying interface for many depth models
    """

    def __init__(
        self,
        model_type,
        num_classes,
        **kwargs
    ):
        super(OtherModelWrapper, self).__init__(
            **kwargs
        )

        self.model_type = model_type
        self.num_classes = num_classes

        if self.model_type == 'DPT_SwinV2_T_256':
            self._model = torch.hub.load("intel-isl/MiDaS", model_type)

            midastransforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = midastransforms.swin256_transform

            def transform_img(img):
                img_frame = img['image']
                transformed_img_tensor = transform(img_frame)
                transformed_img_np = (
                    transformed_img_tensor
                ).squeeze().cpu().numpy()
                return {'image': transformed_img_np}

            self.transform = transform_img
        elif self.model_type == 'DPT_Hybrid':
            self._model = torch.hub.load("intel-isl/MiDaS", model_type)

            midastransforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = midastransforms.swin256_transform

            def transform_img(img):
                img_frame = img['image']
                transformed_img_tensor = transform(img_frame)
                transformed_img_np = (
                    transformed_img_tensor
                ).squeeze().cpu().numpy()
                return {'image': transformed_img_np}

            self.transform = transform_img
        elif self.model_type == 'DPT_Large':
            self._model = torch.hub.load("intel-isl/MiDaS", model_type)

            midastransforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = midastransforms.swin256_transform

            def transform_img(img):
                img_frame = img['image']
                transformed_img_tensor = transform(img_frame)
                transformed_img_np = (
                    transformed_img_tensor
                ).squeeze().cpu().numpy()
                return {'image': transformed_img_np}

            self.transform = transform_img

        elif self.model_type == 'monodepth':
            raise NotImplementedError

        elif self.model_type == 'monodepth2':
            # os.environ["CUDA_VISIBLE_DEVICES"] = ""
            from monodepth2 import monodepth2
            self._model = monodepth2(
                # model_name='mono_1024x320'
                model_name='mono_640x192'
            )

            self.encoder = self._model.encoder
            self.depth_decoder = self._model.depth_decoder

            def identity_transform(x):
                # resized = cv2.resize(x['image'], (1024, 320))
                resized = cv2.resize(x['image'], (640, 192))
                return {'image': resized}

            self.transform = identity_transform

        elif self.model_type == 'manydepth':
            from manydepth import manydepth
            self._model = manydepth(
                model_name='KITTI_HR_1024_320',
                intrinsics_json_path='media/manydepth/intrinsics.json',
            )

            self.encoder = self._model.encoder
            self.depth_decoder = self._model.depth_decoder

            self.pose_enc = self._model.pose_enc
            self.pose_dec = self._model.pose_dec

            def identity_transform(x):
                resized = cv2.resize(x['image'], (1024, 320))
                return {'image': resized}

            self.transform = identity_transform

        elif self.model_type == 'zerodepth':
            self._model = torch.hub.load(
                "TRI-ML/vidar",
                "ZeroDepth",
                pretrained=True,
                trust_repo=True
            )
            self._image_scale_factor = 0.5
            _image_scale_factor = self._image_scale_factor

            def identity_transform(x):
                img = x['image']
                img = cv2.resize(
                    img,
                    (0, 0),
                    fx=_image_scale_factor,
                    fy=_image_scale_factor,
                )
                img = img.transpose(2, 0, 1)
                img = img.astype(np.float32)
                img = img / 255.0
                return {'image': img}

            self.transform = identity_transform

        elif self.model_type == 'packnet':
            self._model = torch.hub.load(
                "TRI-ML/vidar",
                "PackNet",
                pretrained=True,
                trust_repo=True
            )

            def identity_transform(x):
                img = x['image']
                img = cv2.resize(img, (640, 384))
                img = img.transpose(2, 0, 1)
                img = img.astype(np.float32)
                img = img / 255.0
                return {'image': img}

            self.transform = identity_transform
        else:
            raise NotImplementedError

    def forward(self, x):
        device = self.get_device()
        # intrinsics = self.intrinsics.to(device)
        batch_size = 1
        segmentation = torch.zeros(
            (batch_size, self.num_classes, x.shape[2], x.shape[3]),
        ).to(device=device)

        if self.model_type == 'DPT_SwinV2_T_256':
            inv_depth = self._model(x)
        elif self.model_type == 'DPT_Hybrid':
            inv_depth = self._model(x)
        elif self.model_type == 'DPT_Large':
            inv_depth = self._model(x)
        elif self.model_type == 'monodepth':
            raise NotImplementedError

        elif self.model_type == 'monodepth2':
            x_np = x.squeeze().cpu().numpy().astype(np.uint8)
            inv_depth = torch.tensor(
                self._model.eval(x_np)
            )[:, :, 0].unsqueeze(0).to(device=device, dtype=torch.float32)

        elif self.model_type == 'manydepth':
            x_np = x.squeeze().cpu().numpy().astype(np.uint8)
            inv_depth = torch.tensor(
                self._model.eval(x_np, x_np)
            ).unsqueeze(0).to(device=device)

        elif self.model_type == 'zerodepth':
            intrinsics = torch.tensor(
                self.intrinsic_matrix * self._image_scale_factor
            ).unsqueeze(0)
            intrinsics = intrinsics.to(device=device, dtype=torch.float32)
            inv_depth = self._model(x, intrinsics).squeeze(0)
            inv_depth.pow_(-1)

        elif self.model_type == 'packnet':
            inv_depth = self._model(x)[0].squeeze(0)
            inv_depth.pow_(-1)

        else:
            raise NotImplementedError

        return self.get_semantic_occupancy(inv_depth, segmentation)


@torch.no_grad()
def main(args):
    # Print model name
    enablePrint()
    print(f"Model: {args.model_type}")
    blockPrint()
    device = torch.device(args.device)
    model_type = args.model_type
    dataset_name = args.dataset
    base_path = args.base_path

    if "idd" in dataset_name:
        classes = set(level4_basics_to_class.values())
        num_classes = len(classes)
    elif "bdd" in dataset_name:
        num_classes = 3

    # Load net
    net = OtherModelWrapper(
        model_type=model_type,
        num_classes=num_classes,
    )

    net = net.to(device=device)

    enablePrint()
    print(
        "Model Parameters: {:.2f}M".format(
            sum(p.numel() for p in net.parameters()) / 1e6
        )
    )
    blockPrint()

    transforms = net.transform

    dataset = []
    if "idd" in dataset_name:
        train_datasets, val_datasets = get_all_IDD_Depth_Segmentation_datasets(
            transforms,
            level_id=LEVEL4_BASICS_ID,
            level_2_class=level4_basics_to_class,
            # idd_dataset_path=IDD_DATASET_PATH,
        )
        dataset = train_datasets + val_datasets
        classes = set(level4_basics_to_class.values())
        num_classes = len(classes)
        class_2_color = class_2_color_idd
    elif "bdd" in dataset_name:
        dataset = get_bdd_dataset(
            BDD_Depth_Segmentation, transforms, base_path
        )
        num_classes = 3
        class_2_color = class_2_color_bdd

    if args.compile:
        net = torch.compile(net)

    # Set all randomgen seeds
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Select a random set of N images
    N = 10
    total_size = len(dataset)
    total_use = N
    total_discard = total_size - total_use
    dataset, _ = random_split(
        dataset,
        [total_use, total_discard],
        generator=torch.Generator().manual_seed(0),
    )
    # dataset = dataset[:10]

    """
    Save visuals in the format
    /media/visuals
        /{model_type}_{dataset_name}_{version}
            /RGB

            /GT_Depth

            /GT_Seg

            /Pred_Depth

            /Pred_Seg
    """

    # Make dirs if they don't exist
    visual_path = os.path.join(
        "media/visuals",
        f"{model_type}_{dataset_name}",
    )
    os.makedirs(visual_path, exist_ok=True)

    rgb_path = os.path.join(visual_path, "RGB")
    os.makedirs(rgb_path, exist_ok=True)

    gt_depth_path = os.path.join(visual_path, "GT_Depth")
    os.makedirs(gt_depth_path, exist_ok=True)

    gt_seg_path = os.path.join(visual_path, "GT_Seg")
    os.makedirs(gt_seg_path, exist_ok=True)

    pred_depth_path = os.path.join(visual_path, "Pred_Depth")
    os.makedirs(pred_depth_path, exist_ok=True)

    pred_seg_path = os.path.join(visual_path, "Pred_Seg")
    os.makedirs(pred_seg_path, exist_ok=True)

    for index, batch in tqdm(enumerate(dataset), total=len(dataset)):
        # continue
        x, x_raw, mask_disp, y_disp, mask_seg, y_seg = batch
        x = x.to(device=device, dtype=torch.float32)
        y_disp = y_disp.to(device=device, dtype=torch.float32)
        y_seg = y_seg.to(device=device, dtype=torch.float32)
        mask_disp = mask_disp.to(device=device, dtype=torch.bool)
        mask_seg = mask_seg.to(device=device, dtype=torch.bool)

        y_disp_pred, y_seg_pred, points = net(x)

        # File name 000x.png
        file_name = f"{index:04d}.png"

        # Save RGB
        frame_rgb = x_raw.cpu().numpy()[0]
        frame_rgb_path = os.path.join(rgb_path, file_name)
        cv2.imwrite(frame_rgb_path, frame_rgb)

        # Save GT Depth
        frame_disp = y_disp.cpu().numpy()[0]
        frame_disp = (frame_disp - np.min(frame_disp)) / (
            np.max(frame_disp) - np.min(frame_disp)
        )
        frame_disp = cv2.applyColorMap(
            (frame_disp * 255).astype(np.uint8), cv2.COLORMAP_PLASMA
        )
        frame_disp_path = os.path.join(gt_depth_path, file_name)
        cv2.imwrite(frame_disp_path, frame_disp)

        # Save GT Seg
        disp_img_masks = (
            # y_seg_pred[0].permute(1, 2, 0).cpu().numpy()
            y_seg_pred.permute(1, 2, 0).cpu().numpy()
        )
        disp_img_masks_bool = disp_img_masks > 0.5
        disp_img = np.zeros_like(frame_rgb)
        for class_index in range(disp_img_masks_bool.shape[2]):
            class_mask = disp_img_masks_bool[:, :, class_index]
            class_color = class_2_color[class_index]
            # assign color to mask
            disp_img[class_mask] = class_color
        frame_seg_path = os.path.join(gt_seg_path, file_name)
        cv2.imwrite(frame_seg_path, disp_img)
        ####################################################

        # Save Pred Depth
        disp_img = y_disp_pred[0].cpu().numpy()
        disp_img = (disp_img - np.min(disp_img)) / (
            np.max(disp_img) - np.min(disp_img)
        )
        disp_img = cv2.applyColorMap(
            (disp_img * 255).astype(np.uint8), cv2.COLORMAP_PLASMA
        )
        frame_disp_path = os.path.join(pred_depth_path, file_name)
        cv2.imwrite(frame_disp_path, disp_img)

        # Save Pred Seg
        disp_img_masks = (
            # y_seg_pred[0].permute(1, 2, 0).cpu().numpy()
            y_seg_pred.permute(1, 2, 0).cpu().numpy()
        )
        disp_img_masks_bool = disp_img_masks > 0.5
        disp_img = np.zeros_like(frame_rgb)
        for class_index in range(disp_img_masks_bool.shape[2]):
            class_mask = disp_img_masks_bool[:, :, class_index]
            class_color = class_2_color[class_index]
            # assign color to mask
            disp_img[class_mask] = class_color
        frame_seg_path = os.path.join(pred_seg_path, file_name)
        cv2.imwrite(frame_seg_path, disp_img)

    ###################################################
    # Eval Model FPS only on GPU
    if device.type != "cpu":
        frame_count = 50
        start_time = time.time()
        for _ in range(frame_count):
            _ = net(x)
        end_time = time.time()

        enablePrint()
        print(f"FPS: \
{frame_count / (end_time - start_time):.4f} \
({frame_count} frames in \
{(end_time - start_time):.2f} seconds)")
        blockPrint()
    ###################################################

    disp_wrapper = DepthNet(net)
    seg_wrapper = SegNet(net)
    amp = False
    val_set = dataset

    iou = evaluate_seg(seg_wrapper, val_set, device, amp=amp)
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = evaluate_depth(
        disp_wrapper, val_set, device, amp=amp
    )

    # Print metrics
    enablePrint()
    print(f"IOU: {iou:.4f}")
    print(f"ABS_REL: {abs_rel:.4f}")
    print(f"SQ_REL: {sq_rel:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"RMSE_LOG: {rmse_log:.4f}")
    print(f"A1: {a1:.4f}")
    print(f"A2: {a2:.4f}")
    print(f"A3: {a3:.4f}")
    print("="*20)
    blockPrint()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SOccDPT")

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
        choices=other_models,
        required=True,
        help="Model architecture to use",
    )

    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        # default="cpu",
        help="Device to use for training",
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
