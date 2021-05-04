import torch
import torch.nn as nn
from utils import intersection_over_union


class Yololoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_square_error = nn.MSELoss()
        self.BCE = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        self.var_class = 1
        self.var_noObject = 10
        self.var_Obj = 1
        self.var_box = 10

    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1
        noObj = target[..., 0] == 0

        # No object loss
        no_object_loss = self.BCE((predictions[..., 0:1][noObj]), (target[..., 0:1][noObj]),)

        # Object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_predictions = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors],
                                    dim=-1)
        ious = intersection_over_union(box_predictions[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mean_square_error(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # Box coordinates loss
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log(1e-16 + target[..., 3:5] / anchors)
        box_loss = self.mean_square_error(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # Class loss
        class_loss = self.cross_entropy((predictions[..., 5:][obj]), (target[..., 5][obj].long()))

        return (self.var_box * box_loss
                + self.var_Obj * object_loss
                + self.var_noObject * no_object_loss
                + self.var_class * class_loss)
