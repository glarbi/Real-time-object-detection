import os
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from utils import iou_width_height as iou

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YoloDataset(Dataset):
    def __init__(self, csv_file, image_path, label_path, anchors, image_size=416, grid_size=[13, 26, 52], classes=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.label_path = label_path
        self.image_path = image_path
        self.transform = transform
        self.image_size = image_size
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_scale = self.num_anchors // 3
        self.grid_size = grid_size
        self.IOU_threshold = 0.5
        self.classes = classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        labelPath = os.path.join(self.label_path, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=labelPath, delimiter=" ", ndmin=2), 4, axis=1).tolist()  # to shift [class,x,y,h,w] to [x,y,h,w,class]
        imagePath = os.path.join(self.image_path, self.annotations.iloc[index, 0])
        image = np.array(Image.open(imagePath).convert("RGB"))

        if self.transform:
            augment = self.transform(image=image, bboxes=bboxes)
            image = augment["image"]
            bboxes = augment["bboxes"]

        targets = [torch.zeros((self.num_anchors // 3, grid_size, grid_size, 6)) for grid_size in self.grid_size]

        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            bestAnchor = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3

            for anchor in bestAnchor:
                scale_idx = anchor // self.num_anchors_scale
                anchor_on_scale = anchor % self.num_anchors_scale
                grid_size = self.grid_size[scale_idx]
                i, j = int(grid_size * y), int(grid_size * x)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = grid_size * x - j, grid_size * y - i
                    width_cell, height_cell = (width * grid_size, height * grid_size)
                    box_coors = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coors
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                elif not anchor_taken and iou_anchors[anchor] > self.IOU_threshold:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)