import cv2
import numpy as np
import torch
from typing import List

def normalize_vector(v):
    v_mag = torch.norm(v, p=2, dim=1, keepdim=True)  # Compute the magnitude of each vector, p=2 - l2 norm Euclidean
    v_mag = torch.clamp(v_mag, min=1e-8)  # Avoid division by zero
    v_normalized = v / v_mag  # Normalize each vector

    return v_normalized


def cross_product(u, v):
    i_component = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j_component = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k_component = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    cross_prod = torch.cat((i_component.unsqueeze(1), j_component.unsqueeze(1), k_component.unsqueeze(1)), dim=1)

    return cross_prod


def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]  # First 3D vector
    y_raw = poses[:, 3:6]  # Second 3D vector

    x = normalize_vector(x_raw)  # Normalize the first vector
    z = cross_product(x, y_raw)  # Compute the cross product of x and y_raw to get z
    z = normalize_vector(z)  # Normalize z
    y = cross_product(z, x)  # Compute y by crossing z and x

    # Reshape x, y, z to (batch_size, 3, 1) and concatenate them to form rotation matrices
    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)

    rotation_matrix = torch.cat((x, y, z), dim=2)  # Concatenate along the last dimension

    return rotation_matrix

def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    R = rotation_matrices
    sy = torch.sqrt(R[:, 0, 0]**2 + R[:, 1, 0]**2)
    is_singular = sy < 1e-6

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    x_s = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    y_s = torch.atan2(-R[:, 2, 0], sy)
    z_s = torch.zeros_like(z)

    angles = torch.zeros(R.size(0), 3)
    angles[:, 0] = x * (~is_singular) + x_s * is_singular
    angles[:, 1] = y * (~is_singular) + y_s * is_singular
    angles[:, 2] = z * (~is_singular) + z_s * is_singular

    return angles

def draw_cube(image: np.ndarray, yaw: float, pitch: float, roll: float, bbox: List[int], size: int = 150) -> None:
    yaw, pitch, roll = np.radians([-yaw, pitch, roll])
    x_min, y_min, x_max, y_max = bbox
    tdx = int(x_min + (x_max - x_min) * 0.5)
    tdy = int(y_min + (y_max - y_min) * 0.5)
    face_x, face_y = tdx - 0.5 * size, tdy - 0.5 * size

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)

    x1 = int(size * (cos_yaw * cos_roll) + face_x)
    y1 = int(size * (cos_pitch * sin_roll + cos_roll * sin_pitch * sin_yaw) + face_y)
    x2 = int(size * (-cos_yaw * sin_roll) + face_x)
    y2 = int(size * (cos_pitch * cos_roll - sin_pitch * sin_yaw * sin_roll) + face_y)
    x3 = int(size * sin_yaw + face_x)
    y3 = int(size * (-cos_yaw * sin_pitch) + face_y)

    face_x, face_y = int(face_x), int(face_y)
    red, green, blue = (0, 0, 255), (0, 255, 0), (255, 0, 0)

    cv2.line(image, (face_x, face_y), (x1, y1), red, 2)
    cv2.line(image, (face_x, face_y), (x2, y2), red, 2)
    cv2.line(image, (x2, y2), (x2 + x1 - face_x, y2 + y1 - face_y), red, 2)
    cv2.line(image, (x1, y1), (x1 + x2 - face_x, y1 + y2 - face_y), red, 2)

    cv2.line(image, (face_x, face_y), (x3, y3), blue, 2)
    cv2.line(image, (x1, y1), (x1 + x3 - face_x, y1 + y3 - face_y), blue, 2)
    cv2.line(image, (x2, y2), (x2 + x3 - face_x, y2 + y3 - face_y), blue, 2)
    cv2.line(image, (x2 + x1 - face_x, y2 + y1 - face_y), (x3 + x1 + x2 - 2 * face_x, y3 + y2 + y1 - 2 * face_y), blue, 2)

    cv2.line(image, (x3 + x1 - face_x, y3 + y1 - face_y), (x3 + x1 + x2 - 2 * face_x, y3 + y2 + y1 - 2 * face_y), green, 2)
    cv2.line(image, (x2 + x3 - face_x, y2 + y3 - face_y), (x3 + x1 + x2 - 2 * face_x, y3 + y2 + y1 - 2 * face_y), green, 2)
    cv2.line(image, (x3, y3), (x3 + x1 - face_x, y3 + y1 - face_y), green, 2)
    cv2.line(image, (x3, y3), (x3 + x2 - face_x, y3 + y2 - face_y), green, 2)

def draw_axis(image: np.ndarray, yaw: float, pitch: float, roll: float, bbox: List[int], size_ratio: float = 0.5) -> None:
    yaw, pitch, roll = np.radians([-yaw, pitch, roll])
    x_min, y_min, x_max, y_max = bbox
    tdx = int((x_min + x_max) / 2)
    tdy = int((y_min + y_max) / 2)
    size = min(x_max - x_min, y_max - y_min) * size_ratio

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)

    x1 = int(size * (cos_yaw * cos_roll) + tdx)
    y1 = int(size * (cos_pitch * sin_roll + cos_roll * sin_pitch * sin_yaw) + tdy)

    x2 = int(size * (-cos_yaw * sin_roll) + tdx)
    y2 = int(size * (cos_pitch * cos_roll - sin_pitch * sin_yaw * sin_roll) + tdy)

    x3 = int(size * sin_yaw + tdx)
    y3 = int(size * (-cos_yaw * sin_pitch) + tdy)

    cv2.line(image, (tdx, tdy), (x1, y1), (0, 0, 255), 2)
    cv2.line(image, (tdx, tdy), (x2, y2), (0, 255, 0), 2)
    cv2.line(image, (tdx, tdy), (x3, y3), (255, 0, 0), 2)
