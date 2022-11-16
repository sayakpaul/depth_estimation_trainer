# Taken from https://github.com/diode-dataset/diode-devkit/blob/master/metrics.py


import numpy as np


def errors(eval_preds):
    pred, gt = eval_preds

    valid_mask = gt > 0
    pred_eval, gt_eval = pred[valid_mask], gt[valid_mask]

    threshold = np.maximum((gt_eval / pred_eval), (pred_eval / gt_eval))

    delta1 = (threshold < 1.25).mean()
    delta2 = (threshold < 1.25 ** 2).mean()
    delta3 = (threshold < 1.25 ** 3).mean()

    abs_diff = np.abs(pred_eval - gt_eval)

    mae = np.mean(abs_diff)
    rmse = np.sqrt(np.mean(np.power(abs_diff, 2)))
    abs_rel = np.mean(abs_diff / gt_eval)

    log_abs_diff = np.abs(np.log10(pred_eval) - np.log10(gt_eval))

    log_mae = np.mean(log_abs_diff)
    log_rmse = np.sqrt(np.mean(np.power(log_abs_diff, 2)))

    return dict(
        mae=mae,
        rmse=rmse,
        abs_rel=abs_rel,
        log_mae=log_mae,
        log_rmse=log_rmse,
        delta1=delta1,
        delta2=delta2,
        delta3=delta3,
    )
