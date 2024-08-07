import numpy as np
from pyod.models.ecod import ECOD


def is_outlier(points, percentile_thresh=99.9, min_abs_thresh=4):
    """
    Returns the indexes that are outliers based on the modified (median absolute deviation) z-score method,
    as well as their z-scores.

    Filtering is based on a relative cutoff based on a percentile of the z-scores, as well as an absolute minimum cutoff:
    the maximum of the two is used to cutoff the scores.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        percentile_thresh : The percentile of the z-scores to use as a threshold. Observations with a modified z-score (based on the median absolute deviation) greater than the percentile value of z-scores will be classified as outliers.
        min_abs_thresh : The modified z-score to use as a minimum absolute threshold. If the z-score distribution percentile threshold is lower than this value, this value will be used instead.

    Returns:
    --------
        selected_outliers : The indexes of observations that are outliers based on the modified z-score method.
        outlier_scores : The modified z-scores of the observations that are outliers based on the modified z-score method.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    mask = modified_z_score > max(
        np.percentile(modified_z_score, percentile_thresh), min_abs_thresh
    )

    return np.flatnonzero(mask), modified_z_score[mask]


def is_outlier_ecod(points, embeddings):
    embeddings = np.vstack(embeddings)
    model = ECOD(contamination=0.1, n_jobs=1)
    model.fit(embeddings)
    result = model.predict(embeddings)
    pass
