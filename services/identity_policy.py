import numpy as np


def is_same_position(box1, box2):
    """Checks if the center of a new box is very close to an old box."""
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2

    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    allowed_movement_radius = (box1[2] - box1[0]) * 0.6

    return dist < allowed_movement_radius
