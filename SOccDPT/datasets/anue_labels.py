#!/usr/bin/python
#
# AutoNUE labels
#

from collections import namedtuple
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple(
    "Label",
    [
        "name",
        "id",
        "csId",
        "csTrainId",
        "level4Id",
        "level3Id",
        "level2IdName",
        "level2Id",
        "level1Id",
        "hasInstances",
        "ignoreInEval",
        "color",
    ],
)


# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------


labels = [
    Label(
        "road", 0, 7, 0, 0, 0, "drivable", 0, 0, False, False, (128, 64, 128)
    ),
    Label(
        "parking",
        1,
        9,
        255,
        1,
        1,
        "drivable",
        1,
        0,
        False,
        False,
        (250, 170, 160),
    ),
    Label(
        "drivable fallback",
        2,
        255,
        255,
        2,
        1,
        "drivable",
        1,
        0,
        False,
        False,
        (81, 0, 81),
    ),
    Label(
        "sidewalk",
        3,
        8,
        1,
        3,
        2,
        "non-drivable",
        2,
        1,
        False,
        False,
        (244, 35, 232),
    ),
    Label(
        "rail track",
        4,
        10,
        255,
        3,
        3,
        "non-drivable",
        3,
        1,
        False,
        False,
        (230, 150, 140),
    ),
    Label(
        "non-drivable fallback",
        5,
        255,
        9,
        4,
        3,
        "non-drivable",
        3,
        1,
        False,
        False,
        (152, 251, 152),
    ),
    Label(
        "person",
        6,
        24,
        11,
        5,
        4,
        "living-thing",
        4,
        2,
        True,
        False,
        (220, 20, 60),
    ),
    Label(
        "animal",
        7,
        255,
        255,
        6,
        4,
        "living-thing",
        4,
        2,
        True,
        True,
        (246, 198, 145),
    ),
    Label(
        "rider",
        8,
        25,
        12,
        7,
        5,
        "living-thing",
        5,
        2,
        True,
        False,
        (255, 0, 0),
    ),
    Label(
        "motorcycle",
        9,
        32,
        17,
        8,
        6,
        "2-wheeler",
        6,
        3,
        True,
        False,
        (0, 0, 230),
    ),
    Label(
        "bicycle",
        10,
        33,
        18,
        9,
        7,
        "2-wheeler",
        6,
        3,
        True,
        False,
        (119, 11, 32),
    ),
    Label(
        "autorickshaw",
        11,
        255,
        255,
        10,
        8,
        "autorickshaw",
        7,
        3,
        True,
        False,
        (255, 204, 54),
    ),
    Label("car", 12, 26, 13, 11, 9, "car", 7, 3, True, False, (0, 0, 142)),
    Label(
        "truck",
        13,
        27,
        14,
        12,
        10,
        "large-vehicle",
        8,
        3,
        True,
        False,
        (0, 0, 70),
    ),
    Label(
        "bus",
        14,
        28,
        15,
        13,
        11,
        "large-vehicle",
        8,
        3,
        True,
        False,
        (0, 60, 100),
    ),
    Label(
        "caravan",
        15,
        29,
        255,
        14,
        12,
        "large-vehicle",
        8,
        3,
        True,
        True,
        (0, 0, 90),
    ),
    Label(
        "trailer",
        16,
        30,
        255,
        15,
        12,
        "large-vehicle",
        8,
        3,
        True,
        True,
        (0, 0, 110),
    ),
    Label(
        "train",
        17,
        31,
        16,
        15,
        12,
        "large-vehicle",
        8,
        3,
        True,
        True,
        (0, 80, 100),
    ),
    Label(
        "vehicle fallback",
        18,
        355,
        255,
        15,
        12,
        "large-vehicle",
        8,
        3,
        True,
        False,
        (136, 143, 153),
    ),
    Label(
        "curb",
        19,
        255,
        255,
        16,
        13,
        "barrier",
        9,
        4,
        False,
        False,
        (220, 190, 40),
    ),
    Label(
        "wall",
        20,
        12,
        3,
        17,
        14,
        "barrier",
        9,
        4,
        False,
        False,
        (102, 102, 156),
    ),
    Label(
        "fence",
        21,
        13,
        4,
        18,
        15,
        "barrier",
        10,
        4,
        False,
        False,
        (190, 153, 153),
    ),
    Label(
        "guard rail",
        22,
        14,
        255,
        19,
        16,
        "barrier",
        10,
        4,
        False,
        False,
        (180, 165, 180),
    ),
    Label(
        "billboard",
        23,
        255,
        255,
        20,
        17,
        "structures",
        11,
        4,
        False,
        False,
        (174, 64, 67),
    ),
    Label(
        "traffic sign",
        24,
        20,
        7,
        21,
        18,
        "structures",
        11,
        4,
        False,
        False,
        (220, 220, 0),
    ),
    Label(
        "traffic light",
        25,
        19,
        6,
        22,
        19,
        "structures",
        11,
        4,
        False,
        False,
        (250, 170, 30),
    ),
    Label(
        "pole",
        26,
        17,
        5,
        23,
        20,
        "structures",
        12,
        4,
        False,
        False,
        (153, 153, 153),
    ),
    Label(
        "polegroup",
        27,
        18,
        255,
        23,
        20,
        "structures",
        12,
        4,
        False,
        False,
        (153, 153, 153),
    ),
    Label(
        "obs-str-bar-fallback",
        28,
        255,
        255,
        24,
        21,
        "structures",
        12,
        4,
        False,
        False,
        (169, 187, 214),
    ),
    Label(
        "building",
        29,
        11,
        2,
        25,
        22,
        "construction",
        13,
        5,
        False,
        False,
        (70, 70, 70),
    ),
    Label(
        "bridge",
        30,
        15,
        255,
        26,
        23,
        "construction",
        13,
        5,
        False,
        False,
        (150, 100, 100),
    ),
    Label(
        "tunnel",
        31,
        16,
        255,
        26,
        23,
        "construction",
        13,
        5,
        False,
        False,
        (150, 120, 90),
    ),
    Label(
        "vegetation",
        32,
        21,
        8,
        27,
        24,
        "vegetation",
        14,
        5,
        False,
        False,
        (107, 142, 35),
    ),
    Label(
        "sky", 33, 23, 10, 28, 25, "sky", 15, 6, False, False, (70, 130, 180)
    ),
    Label(
        "fallback background",
        34,
        255,
        255,
        29,
        25,
        "object fallback",
        15,
        6,
        False,
        False,
        (169, 187, 214),
    ),
    Label(
        "unlabeled",
        35,
        0,
        255,
        255,
        255,
        "void",
        255,
        255,
        False,
        True,
        (0, 0, 0),
    ),
    Label(
        "ego vehicle",
        36,
        1,
        255,
        255,
        255,
        "void",
        255,
        255,
        False,
        True,
        (0, 0, 0),
    ),
    Label(
        "rectification border",
        37,
        2,
        255,
        255,
        255,
        "void",
        255,
        255,
        False,
        True,
        (0, 0, 0),
    ),
    Label(
        "out of roi",
        38,
        3,
        255,
        255,
        255,
        "void",
        255,
        255,
        False,
        True,
        (0, 0, 0),
    ),
    Label(
        "license plate",
        39,
        255,
        255,
        255,
        255,
        "vehicle",
        255,
        255,
        False,
        True,
        (0, 0, 142),
    ),
]

level3_to_class = {}
for i in range(0, 25 + 1, 1):
    level3_to_class[i] = i
level3_to_class[255] = 26

level3_to_color = {}
for label in labels:
    class_id = level3_to_class[label.level3Id]
    level3_to_color[class_id] = label.color

class_to_level3 = {}
for k, v in level3_to_class.items():
    class_to_level3[v] = k

level1_to_class = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    255: 7,
}

class_to_level1 = {}
for k, v in level1_to_class.items():
    class_to_level1[v] = k

level1_to_level3 = {}
for label in labels:
    level1_to_level3[label.level1Id] = label.level3Id

level1_to_color = {
    0: (127, 127, 127),
    1: (0, 0, 0),
    2: (255, 0, 0),
    3: (10, 10, 255),
    4: (80, 80, 80),
    5: (0, 255, 0),
    6: (10, 10, 0),
    7: (0, 0, 255),
}

# Level1_road to class
level1_road_to_class = {}
level1_road_to_color = {
    0: (128, 64, 128),
    1: (244, 35, 232),
}
for label in labels:
    if label.level2IdName == "drivable":
        level1_road_to_class[label.level1Id] = 0
    elif label.level2IdName == "non-drivable":
        level1_road_to_class[label.level1Id] = 1


# Level4_road to class
level4_road_to_class = {
    0: 0,  # road
    1: 1,  # parking
    2: 2,  # drivable fallback
    3: 3,  # sidewalk, rail track
    4: 4,  # non-drivable fallback
}
level4_road_to_color = {
    0: (128, 64, 128),  # road
    1: (250, 170, 160),  # parking
    2: (81, 0, 81),  # drivable fallback
    3: (244, 35, 232),  # (230,150,140),  # sidewalk, rail track
    4: (152, 251, 152),  # non-drivable fallback
}

# --------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
# --------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label = {label.name: label for label in labels}
# id to label object
id2label = {label.id: label for label in labels}


def assureSingleInstanceName(name):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[: -len("group")]
    # test if the new name exists
    if name not in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name


class IDD_Dataset:
    def __init__(
        self,
        leftImg8bit_path="~/Datasets/IDD_Segmentation/leftImg8bit/train/0/",
        gtFine_path="~/Datasets/IDD_Segmentation/gtFine/train/0/",
        depth_path="~/Datasets/IDD_Segmentation/depth/train/0/",
        level_id="level1Ids",
        level_2_class=level1_to_class,
    ) -> None:
        self.leftImg8bit_path = leftImg8bit_path
        self.gtFine_path = gtFine_path
        self.depth_path = depth_path

        self.level_id = level_id
        self.level_2_class = level_2_class
        # self.num_classes = len(self.level_2_class.keys())
        self.classes = set(self.level_2_class.values())
        self.num_classes = len(self.classes)

        # check if all are directories
        assert os.path.isdir(leftImg8bit_path), (
            "leftImg8bit_path is not a directory " + leftImg8bit_path
        )
        assert os.path.isdir(gtFine_path), (
            "gtFine_path is not a directory " + gtFine_path
        )
        assert os.path.isdir(depth_path), (
            "depth_path is not a directory " + depth_path
        )

        self.files = glob.glob(os.path.join(leftImg8bit_path, "*.png"))
        self.files = [
            os.path.basename(f).replace("_leftImg8bit.png", "")
            for f in self.files
        ]

        self.leftImg8bit_files = [
            os.path.join(leftImg8bit_path, f + "_leftImg8bit.png")
            for f in self.files
        ]
        self.gtFine_files = [
            os.path.join(
                gtFine_path, f + "_gtFine_label{}.png".format(self.level_id)
            )
            for f in self.files
        ]
        self.depth_files = [
            os.path.join(depth_path, f + "_depth.png") for f in self.files
        ]

        for i in range(len(self.files)):
            assert os.path.isfile(self.leftImg8bit_files[i]), (
                "File not found: " + self.leftImg8bit_files[i]
            )
            assert os.path.isfile(self.gtFine_files[i]), (
                "File not found: " + self.gtFine_files[i]
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        leftImg8bit = cv2.imread(self.leftImg8bit_files[index])
        gtFine = cv2.imread(self.gtFine_files[index])
        # depth = cv2.imread(self.depth_files[index])
        depth = np.zeros(
            (leftImg8bit.shape[0], leftImg8bit.shape[1]), dtype=np.uint8
        )

        leftImg8bit = cv2.resize(leftImg8bit, (1920, 1080))
        gtFine = cv2.resize(gtFine, (1920, 1080))
        depth = cv2.resize(depth, (1920, 1080))

        gtFine = cv2.cvtColor(gtFine, cv2.COLOR_BGR2GRAY)

        seg_map = np.zeros(
            (gtFine.shape[0], gtFine.shape[1], self.num_classes), dtype=bool
        )
        for class_id in self.level_2_class:
            seg_map[:, :, self.level_2_class[class_id]] = gtFine == class_id

        return leftImg8bit, seg_map, depth


def color_mask(seg_map, color_map=level1_to_color):
    img = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for class_id in color_map:
        img[seg_map[:, :, class_id]] = color_map[class_id]
    return img


IDD_DATASET_PATH = os.path.expanduser("~/Datasets/IDD_Segmentation/")


def get_train_val_test_folders(dataset_path=IDD_DATASET_PATH):
    assert os.path.isdir(dataset_path), "dataset_path is not a directory"
    train_folders = glob.glob(
        os.path.join(dataset_path, "leftImg8bit/train/*")
    )
    train_folders = [os.path.basename(f) for f in train_folders]
    train_folders.sort()

    val_folders = glob.glob(os.path.join(dataset_path, "leftImg8bit/val/*"))
    val_folders = [os.path.basename(f) for f in val_folders]
    val_folders.sort()

    test_folders = glob.glob(os.path.join(dataset_path, "leftImg8bit/test/*"))
    test_folders = [os.path.basename(f) for f in test_folders]
    test_folders.sort()

    return train_folders, val_folders, test_folders


# --------------------------------------------------------------------------------
# Main for testing
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    # train_folders, val_folders, test_folders = get_train_val_test_folders()

    # print(train_folders)
    # print(val_folders)
    # print(test_folders)
    # exit()
    plt.ion()
    dataset = IDD_Dataset()

    for i in range(len(dataset)):
        leftImg8bit, gtFine, depth = dataset[i]
        gtFine_color = color_mask(gtFine)
        vis_img = np.concatenate((leftImg8bit, gtFine_color), axis=1)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        print(
            leftImg8bit.shape,
            gtFine.shape,
        )

        # plot using plt
        plt.imshow(vis_img)
        plt.show()

        # waitkey plt
        plt.pause(0.1)
