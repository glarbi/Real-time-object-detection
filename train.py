import torch
import config
import warnings
from utils import (
    mean_average_precision,
    get_evaluation_bboxes,
    check_class_accuracy,
    save_checkpoint,
    load_checkpoint,
    get_loaders
)
from tqdm import tqdm
from model import Yolo
import torch.optim as optim
from loss import Yololoss

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

def train(train_loader, model, optimizer, lossfunction, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batchIdx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = ( y[0].to(config.DEVICE),
                       y[1].to(config.DEVICE),
                       y[2].to(config.DEVICE))
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                lossfunction(out[0], y0, scaled_anchors[0]) +
                lossfunction(out[1], y1, scaled_anchors[1]) +
                lossfunction(out[2], y2, scaled_anchors[2])
            )
        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss = mean_loss)

def main():
    model = Yolo(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    lossfunction = Yololoss()
    scaler = torch.cuda.amp.GradScaler()
    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path = config.DATASET + "/train.csv", test_csv_path = config.DATASET + "/test.csv")

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        )

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.GRID_SIZE).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train(train_loader, model, optimizer, lossfunction, scaler, scaled_anchors)
        if config.SAVE_MODEL:
            save_checkpoint(model, optimizer, filename=config.CHECKPOINT_FILE)
        if epoch > 0 and epoch % 3 == 0:
            check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            model.train()

main()