import torch as t
import torch.nn as nn
import torch.nn.functional as F


class RPNLoss(nn.Module):
    def __init__(self):
        super(RPNLoss, self).__init__()

    def compute_cls_loss(self, rpn_cls_t, rpn_cls_logits):
        # rpn_cls_t: [bs, 576, 1]
        # rpn_cls_logits: [bs, 576, 2]
        rpn_cls = t.squeeze(rpn_cls_t, dim=-1)
        valid_logits = rpn_cls_logits[rpn_cls != -1]
        valid_rpn_cls = rpn_cls[rpn_cls != -1]
        loss = F.cross_entropy(valid_logits, valid_rpn_cls)

        if loss.size() == 0:
            return t.zeros(size=1).double()
        else:
            return loss

    def batch_pack(self, target_bbox, batch_counts, batch):
        outputs = []

        for i in range(batch):
            outputs.append(target_bbox[i, :batch_counts])

        return t.cat(outputs, dim=0)

    def compute_bbox_loss(self, target_bbox, rpn_truth, pre_bbox):
        batch = rpn_truth.shape[0]
        rpn_cls = t.squeeze(rpn_truth, dim=-1)
        rpn_bbox = pre_bbox[rpn_cls == 1]
        batch_counts = t.sum((pre_bbox == 1).int(), dim=1)
        target_bbox = self.batch_pack(target_bbox, batch_counts, batch)
        diff = t.abs(target_bbox - pre_bbox)
        loss = diff[diff < 1.0] ** 2 * 0.5 + (1 - diff[diff < 1.0]) * (diff - 0.5)

        if loss.size() == 0:
            return t.zeros(size=1).double()
        else:
            return loss

    def forward(self, rpn_truth, rpn_class_logits, target_box, pre_bbox):
        cls_loss = self.compute_cls_loss(rpn_truth, rpn_class_logits)
        bbox_loss = self.compute_bbox_loss(target_box, rpn_truth, pre_bbox)


if __name__ == "__main__":
    rpnloss = RPNLoss()
    cls_t = t.randint(-1, 2, size=(1, 576, 1))
    logits = t.randn((1, 576, 2))
    logits = F.softmax(logits, dim=-1)
    ls = rpnloss(cls_t, logits)


