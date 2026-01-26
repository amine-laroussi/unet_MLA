import numpy as np
from collections import Counter
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries

#------------------------------------------
# pixel error metric
#------------------------------------------
def pixel_error (pred, gt,ignore_value=255):
    valid= gt!=ignore_value
    return np.mean (pred[valid]!=gt[valid])

#------------------------------------------
# rand error metric
#------------------------------------------
def _comb2(n):
    """
    Number of unordered pairs in a set of size n.
    """
    return n * (n - 1) // 2

def rand_error(pred, gt, ignore_value=255):
    valid = gt != ignore_value
    pred = pred[valid].ravel()
    gt   = gt[valid].ravel()

    n_pixels = len(gt)
    if n_pixels < 2:
        return 0.0


    gt_counts   = Counter(gt)
    pred_counts = Counter(pred)

    sum_gt = sum(_comb2(c) for c in gt_counts.values())


    sum_pred = sum(_comb2(c) for c in pred_counts.values())


    intersection = Counter(zip(gt, pred))
    sum_intersection = sum(_comb2(c) for c in intersection.values())

    total_pairs = _comb2(n_pixels)

    true_positive = sum_intersection
    false_positive = sum_pred - sum_intersection
    false_negative = sum_gt - sum_intersection
    true_negative = total_pairs - true_positive - false_positive - false_negative

    rand_index = (
        true_positive + true_negative
    ) / total_pairs

    return 1.0 - rand_index

#------------------------------------------
# warping error metric
#------------------------------------------
def warping_error_bidirectional(pred, gt):
        # Extract boundaries
    gt_bound = find_boundaries(gt, mode='inner')
    pred_bound = find_boundaries(pred, mode='inner')

    if gt_bound.sum() == 0 or pred_bound.sum() == 0:
        return np.nan

    # Distance transforms
    dist_to_pred = distance_transform_edt(~pred_bound)
    dist_to_gt   = distance_transform_edt(~gt_bound)

    # Distances in both directions
    d_gt_to_pred = dist_to_pred[gt_bound]
    d_pred_to_gt = dist_to_gt[pred_bound]

    # Symmetric boundary distance
    return 0.5 * (d_gt_to_pred.mean() + d_pred_to_gt.mean())



