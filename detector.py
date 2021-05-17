
import cv2
import torch
import config
import numpy as np
from model import Yolo
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from torchvision import transforms
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import (
    intersection_over_union,
    non_max_suppression,
    cells_to_bboxes
)

def draw_box(frame, aff, boxes):

    cmap = plt.get_cmap("tab20b")
    class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    height, width, _ = aff.shape
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upperleftx = box[0] - box[2] / 2
        upperlefty = box[1] - box[3] / 2
        bottomrightx = box[0] + box[2] / 2
        bottomrighty = box[1] + box[3] / 2
        label = class_labels[int(class_pred)]
        p1 = tuple([int(upperleftx*width), int(upperlefty*height)])
        p2 = tuple([int(bottomrightx*width), int(bottomrighty*height)])
        color = colors[int(class_pred)]
        cv2.rectangle(aff, p1, p2, color, 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
        p3 = (p1[0], p1[1] - text_size[1] - 4)
        p4 = (p1[0] + text_size[0] + 4, p1[1])
        cv2.rectangle(aff, p3, p4, color, -1)
        cv2.putText(aff, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1)
    out.write(aff)
    

    
def get_bboxes(model, frame, aff, thresh, iou_thresh, anchors):
    with torch.no_grad():
        out = model(frame[None, ...])
        bboxes = [[] for _ in range(1)]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i].clone().detach().to(config.DEVICE)
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
        model.train()
    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        if nms_boxes : draw_box(frame, aff, nms_boxes)

print('Loading network...')
model = Yolo(num_classes=config.NUM_CLASSES).to(config.DEVICE)
print('Network loaded')

print("=> Loading checkpoint")
checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
model.load_state_dict(checkpoint["state_dict"])

model.eval()
cap = cv2.VideoCapture('testVideo.mp4')
output_path = 'detectionvideo.mp4'
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
read_frames = 0


start_time = datetime.now()
print('Detecting...')
while cap.isOpened():
    retflag, frame = cap.read()
    if frame is None:
        break
    aff = frame.copy()
    read_frames += 1
    if retflag:
        test_transforms = A.Compose(
            [
                A.LongestMaxSize(max_size=config.IMAGE_SIZE),
                A.PadIfNeeded(
                    min_height=config.IMAGE_SIZE, min_width=config.IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
                ),
                A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
                ToTensorV2(),
            ]
        )

        augment = test_transforms(image=frame)
        frame = augment["image"]
        frame = frame.to(config.DEVICE)
        

    S = [13, 26, 52]
    anchors = config.ANCHORS
    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    get_bboxes(model, frame, aff, config.CONF_THRESHOLD, config.MAP_IOU_THRESH, scaled_anchors)

end_time = datetime.now()
print('Detection finished in %s' % (end_time - start_time))
print('Total frames:', read_frames)
cap.release()
out.release()