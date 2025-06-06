import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def ellipse_points(n_points, radius_x, radius_y, center=(0.5, 0.5), angle_offset=0.0):
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False) + angle_offset
    x = center[0] + radius_x * np.cos(angles)
    y = center[1] + radius_y * np.sin(angles)
    return np.vstack((x, y)).T

def balanced_min_distance(points, bounds=(0, 1)):
    dist = euclidean_distances(points)
    np.fill_diagonal(dist, np.inf)
    min_point_dist = np.min(dist) / 2
    lower, upper = bounds
    min_bound_dist = np.min(np.hstack([
        points[:, 0] - lower,
        upper - points[:, 0],
        points[:, 1] - lower,
        upper - points[:, 1]
    ]))
    return min(min_point_dist, min_bound_dist)

def shrink_ellipse_to_balance(points, center=(0.5, 0.5), bounds=(0, 1), shrink_step=0.001, max_iter=1000):
    best_points = points.copy()
    best_balance = balanced_min_distance(points, bounds)
    for i in range(max_iter):
        factor = 1.0 - shrink_step * (i + 1)
        if factor <= 0:
            break
        shrunk_points = center + factor * (points - center)
        balance = balanced_min_distance(shrunk_points, bounds)
        if balance > best_balance:
            best_balance = balance
            best_points = shrunk_points
        else:
            break
    return best_points, best_balance

def init_points_with_offsets(n_points, K_offsets=1, bounds=(0, 1)):
    center = ((bounds[1] + bounds[0]) / 2, (bounds[1] + bounds[0]) / 2)
    radius = (bounds[1] - bounds[0]) / 2
    angular_gap = 2 * np.pi / n_points

    results = []
    best_points = None
    best_balance = -np.inf

    for k in range(K_offsets):
        angle_offset = angular_gap * k / K_offsets
        initial = ellipse_points(n_points, radius, radius, center=center, angle_offset=angle_offset)
        shrunk, balance = shrink_ellipse_to_balance(initial, center=center, bounds=bounds)
        final_radius = np.linalg.norm(shrunk[0] - center)
        results.append((shrunk, balance, center, final_radius))
        if balance > best_balance:
            best_balance = balance
            best_points = shrunk

    return results, best_points, best_balance
