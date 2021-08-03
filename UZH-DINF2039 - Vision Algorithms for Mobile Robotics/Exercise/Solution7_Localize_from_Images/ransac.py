import os
import numpy as np
from matplotlib import pyplot as plt
from functools import partial


def ransac(points, fit_fn, eval_fn, refine_fn=None,
           n_iters=100, thresh=0.2, inlier_ratio=None, points_per_step=3):
    """
    Implement RANSAC algorithm for robust model fitting.
    Parameters:
        points: [(N, ...), ..., (N, ...)] list of arrays.
        fit_fn: The function to perform model fitting on a number of points.
        eval_fn: The function to evaluate fitted model on a number of points.
    """
    n_points = points[0].shape[0]
    if refine_fn is None:
        refine_fn = fit_fn

    inlier_ids = np.array([], dtype=float)
    # Iterate
    for iter in range(n_iters):
        # Fit a model with randomly selected points
        select_ids = np.random.choice(n_points, points_per_step)
        points_select = [sub_points[select_ids] for sub_points in points]
        model_est = fit_fn(points_select)
        # Count inliers according to the fitted model
        error = eval_fn(points, model_est)
        inlier_ids_est = np.where(error < thresh)[0]

        if inlier_ids_est.shape[0] > inlier_ids.shape[0]:
            inlier_ids = inlier_ids_est
            inliers = [sub_points[inlier_ids] for sub_points in points]
            # Refine the best model with all inliers
            model = refine_fn(inliers)
        if inlier_ratio is not None:
            if inlier_ids.shape[0] > n_points*inlier_ratio:
                break
    return model, inlier_ids, iter+1


def adaptive_ransac(points, fit_fn, eval_fn, refine_fn=None,
                    n_iters=100, thresh=0.2, points_per_step=3):
    """
    Implement adaptive RANSAC algorithm for robust and fast model fitting.
    Both inlier ratio and iterations (practically needed) are dymanically estimated.
    Parameters:
        points: [(N, ...), ..., (N, ...)] list of arrays.
        fit_fn: The function to perform model fitting on a number of points.
        eval_fn: The function to evaluate fitted model on a number of points.
    """
    n_points = points[0].shape[0]
    if refine_fn is None:
        refine_fn = fit_fn

    inlier_ids = np.array([], dtype=float)
    # Iterate
    for iter in range(n_iters):
        # Fit a model with randomly selected points
        select_ids = np.random.choice(n_points, points_per_step)
        points_select = [sub_points[select_ids] for sub_points in points]
        model_est = fit_fn(points_select)
        # Count inliers according to the fitted model
        error = eval_fn(points, model_est)
        inlier_ids_est = np.where(error < thresh)[0]

        if inlier_ids_est.shape[0] > inlier_ids.shape[0]:
            inlier_ids = inlier_ids_est
            inliers = [sub_points[inlier_ids] for sub_points in points]
            # Refine the best model with all inliers
            model = refine_fn(inliers)

        # Estimate inlier ratio
        inlier_ratio = inlier_ids.shape[0] / n_points
        max_iter = int(np.log(0.01) / np.log(1 - inlier_ratio**points_per_step))
        if iter >= max_iter:
            break
    return model, inlier_ids, iter+1


def polyfit(points, deg=2):
    points = points[0]
    poly = np.polyfit(points[:, 0], points[:, 1], deg=deg)
    return poly


def polyfit_rms(points, poly_est):
    points = points[0]
    f_poly_est = np.poly1d(poly_est)
    points_x, points_y = points[:, 0], points[:, 1]
    points_y_est = f_poly_est(points_x)
    rms_error = np.sqrt(np.square(points_y - points_y_est))
    return rms_error


if __name__ == '__main__':
    np.random.seed(10)
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)

    # Construct test samples
    n_inliers = 20
    n_outliers = 10
    noise_ratio = 0.1
    poly = np.random.uniform(size=3)
    f_poly = np.poly1d(poly)

    x_extremum = - 0.5 * poly[1] / (poly[0]+1e-12)
    x_min = x_extremum - 0.5
    y_min = f_poly(x_extremum)
    y_max = f_poly(x_min)
    y_range = y_max - y_min     # x_range = 1.0
    noise_range = y_range * noise_ratio

    inliers_x = np.random.uniform(size=n_inliers) + x_min
    inliers_y = f_poly(inliers_x) + (np.random.uniform(size=n_inliers) * 2 - 1) * noise_range
    inliers = np.stack((inliers_x, inliers_y), axis=1)
    outliers_x = np.random.uniform(size=n_outliers) + x_min
    outliers_y = np.random.uniform(size=n_outliers) * y_range + y_min
    outliers = np.stack((outliers_x, outliers_y), axis=1)
    points = np.concatenate((inliers, outliers), axis=0)

    n_iters = 100
    inlier_ratio = 0.66     # A prior for inlier ratio
    points_per_step = 3

    # polynomial fitting with all points
    poly_est = polyfit([points], deg=2)
    f_poly_est = np.poly1d(poly_est)

    # polynomial fitting with RANSAC
    poly_est_ransac, _, _ = ransac([points], fit_fn=partial(polyfit, deg=2), eval_fn=polyfit_rms,
                                    n_iters=n_iters, thresh=noise_range,
                                    inlier_ratio=inlier_ratio, points_per_step=points_per_step)
    f_poly_est_ransac = np.poly1d(poly_est_ransac)

    # Visualize the groundtruth and estimated polynomial
    fig = plt.figure()
    x = np.arange(x_min, x_min+1, 0.01)
    y_gt = f_poly(x)
    plt.plot(x, y_gt, color='blue')
    y_est = f_poly_est(x)
    plt.plot(x, y_est, color='red')
    y_est_ransac = f_poly_est_ransac(x)
    plt.plot(x, y_est_ransac, color='green')
    plt.scatter(points[:, 0], points[:, 1])
    fig.savefig(os.path.join(save_dir, 'ransac_polynomial.png'))