from torchmetrics import (
    MetricCollection,
    Accuracy,
    Dice,
    JaccardIndex,
    Specificity,
    Recall,
    Precision,
    AUC,
    AUROC,
    ROC,
    F1Score,
)
import numpy as np
import torch


def get_metrics(config, n_classes, mode="train", dist_sync_on_step=False, segmentation=False):
    if segmentation:
        return get_segmentation_metrics(config, n_classes, mode, dist_sync_on_step)
    return get_classification_metrics(config, n_classes, mode, dist_sync_on_step)


def get_classification_metrics(config, n_classes, mode="train", dist_sync_on_step=False):
    if n_classes > 2:
        mdmc_reduce = config.metrics.mdmc_reduce
    else:
        mdmc_reduce = None

    if n_classes in [1, 2]:
        metrics = MetricCollection(
            [
                Accuracy(threshold=config.metrics.threshold, num_classes=1, mdmc_reduce=mdmc_reduce),
                Recall(threshold=config.metrics.threshold, num_classes=1, mdmc_reduce=mdmc_reduce),
                Specificity(threshold=config.metrics.threshold, num_classes=1, mdmc_reduce=mdmc_reduce),
                Precision(threshold=config.metrics.threshold, num_classes=1, mdmc_reduce=mdmc_reduce),
                F1Score(threshold=config.metrics.threshold, num_classes=1, mdmc_reduce=mdmc_reduce),
                AUROC(num_classes=None, mdmc_reduce=mdmc_reduce)
            ]
        )
    else:
        metrics = [
            MetricCollection(
                [
                    Accuracy(
                        average="micro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                    Recall(
                        average="micro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                    Specificity(
                        average="micro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                    Precision(
                        average="micro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                    F1Score(
                        average="micro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                ],
                postfix="_micro",
            ),
            MetricCollection(
                [
                    Accuracy(
                        average="macro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                    Recall(
                        average="macro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                    Specificity(
                        average="macro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                    Precision(
                        average="macro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                    F1Score(
                        average="macro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                ],
                postfix="_macro",
            ),
        ]

        if mode in ["eval", "test"]:
            metrics.append(
                MetricCollection(
                    [
                        Accuracy(
                            average="none",
                            num_classes=1 if n_classes in [1,2] else n_classes,
                            mdmc_reduce=mdmc_reduce,
                            dist_sync_on_step=dist_sync_on_step,
                        ),
                        Recall(
                            average="none",
                            num_classes=1 if n_classes in [1,2] else n_classes,
                            mdmc_reduce=mdmc_reduce,
                            dist_sync_on_step=dist_sync_on_step,
                        ),
                        Specificity(
                            average="none",
                            num_classes=1 if n_classes in [1,2] else n_classes,
                            mdmc_reduce=mdmc_reduce,
                            dist_sync_on_step=dist_sync_on_step,
                        ),
                        Precision(
                            average="none",
                            num_classes=1 if n_classes in [1,2] else n_classes,
                            mdmc_reduce=mdmc_reduce,
                            dist_sync_on_step=dist_sync_on_step,
                        ),
                        F1Score(
                            average="none",
                            num_classes=1 if n_classes in [1,2] else n_classes,
                            mdmc_reduce=mdmc_reduce,
                            dist_sync_on_step=dist_sync_on_step,
                        ),
                    ]
                )
            )
        metrics = MetricCollection(metrics)

    return metrics


def get_segmentation_metrics(config, n_classes, mode="train", dist_sync_on_step=False):
    if n_classes > 1:
        mdmc_reduce = config.metrics.mdmc_reduce
    else:
        mdmc_reduce = None

    if n_classes in [1, 2]:
        metrics = MetricCollection(
            [
                Accuracy(num_classes=1, mdmc_reduce=mdmc_reduce),
                Dice(num_classes=1, mdmc_reduce=mdmc_reduce),
                JaccardIndex(num_classes=2, mdmc_reduce=mdmc_reduce),
            ]
        )
    else:
        metrics = [
            MetricCollection(
                [
                    Accuracy(
                        average="micro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                    Dice(
                        average="micro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                    JaccardIndex(
                        average="micro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                ],
                postfix="_micro",
            ),
            MetricCollection(
                [
                    Accuracy(
                        average="macro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                    Dice(
                        average="macro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                    JaccardIndex(
                        average="macro",
                        num_classes=n_classes,
                        mdmc_reduce=mdmc_reduce,
                        dist_sync_on_step=dist_sync_on_step,
                    ),
                ],
                postfix="_macro",
            ),
        ]

        if mode in ["eval", "test"]:
            metrics.append(
                MetricCollection(
                    [
                        Accuracy(
                            average="none",
                            num_classes=n_classes,
                            mdmc_reduce=mdmc_reduce,
                            dist_sync_on_step=dist_sync_on_step,
                        ),
                        Dice(
                            average="none",
                            num_classes=n_classes,
                            mdmc_reduce=mdmc_reduce,
                            dist_sync_on_step=dist_sync_on_step,
                        ),
                        JaccardIndex(
                            average="none",
                            num_classes=n_classes,
                            mdmc_reduce=mdmc_reduce,
                            dist_sync_on_step=dist_sync_on_step,
                        ),
                    ]
                )
            )
        metrics = MetricCollection(metrics)

    return metrics


# Originally written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class**2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Overall Acc": acc,
        "Mean Acc": acc_cls,
        "FreqW Acc": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }


# SR : Segmentation Result
# GT : Ground Truth


def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR == 1) + (GT == 1)) == 2
    FN = ((SR == 0) + (GT == 1)) == 2

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0) + (GT == 0)) == 2
    FP = ((SR == 1) + (GT == 0)) == 2

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1) + (GT == 1)) == 2
    FP = ((SR == 1) + (GT == 0)) == 2

    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR + GT) == 2)
    Union = torch.sum((SR + GT) >= 1)

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR + GT) == 2)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC
