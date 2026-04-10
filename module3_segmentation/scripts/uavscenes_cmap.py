"""
UAVScenes semantic label ids and RGB palette (camera / image segmentation).

Source: https://github.com/sijieaaa/UAVScenes/blob/main/cmap.py
Use 26 classes (ids 0–25). Pixels not matching any palette entry should use
ignore_label=255 in training/evaluation.
"""

from __future__ import annotations

# fmt: off
CMAP: dict[int, dict] = {
    0: {"name": "background", "RGB": [0, 0, 0]},
    1: {"name": "roof", "RGB": [119, 11, 32]},
    2: {"name": "dirt_motor_road", "RGB": [180, 165, 180]},
    3: {"name": "paved_motor_road", "RGB": [128, 64, 128]},
    4: {"name": "river", "RGB": [173, 216, 230]},
    5: {"name": "pool", "RGB": [0, 80, 100]},
    6: {"name": "bridge", "RGB": [150, 100, 100]},
    7: {"name": "class_7", "RGB": [150, 120, 90]},
    8: {"name": "class_8", "RGB": [70, 70, 70]},
    9: {"name": "container", "RGB": [250, 170, 30]},
    10: {"name": "airstrip", "RGB": [81, 0, 81]},
    11: {"name": "traffic_barrier", "RGB": [102, 102, 156]},
    12: {"name": "class_12", "RGB": [190, 153, 153]},
    13: {"name": "green_field", "RGB": [107, 142, 35]},
    14: {"name": "wild_field", "RGB": [210, 180, 140]},
    15: {"name": "solar_board", "RGB": [220, 220, 0]},
    16: {"name": "umbrella", "RGB": [153, 153, 153]},
    17: {"name": "transparent_roof", "RGB": [0, 0, 90]},
    18: {"name": "car_park", "RGB": [250, 170, 160]},
    19: {"name": "paved_walk", "RGB": [244, 35, 232]},
    20: {"name": "sedan", "RGB": [0, 0, 142]},
    21: {"name": "class_21", "RGB": [224, 224, 192]},
    22: {"name": "class_22", "RGB": [220, 20, 60]},
    23: {"name": "class_23", "RGB": [192, 64, 128]},
    24: {"name": "truck", "RGB": [0, 0, 70]},
    25: {"name": "class_25", "RGB": [0, 60, 100]},
}
# fmt: on

NUM_UAVSCENES_CLASSES: int = len(CMAP)
IGNORE_LABEL: int = 255


def rgb_tuple_to_class_id() -> dict[tuple[int, int, int], int]:
    return {tuple(v["RGB"]): k for k, v in CMAP.items()}


def palette_uavscenes() -> "numpy.ndarray":
    import numpy as np

    return np.array([CMAP[i]["RGB"] for i in range(NUM_UAVSCENES_CLASSES)], dtype=np.uint8)
