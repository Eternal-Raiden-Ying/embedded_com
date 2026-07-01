import numpy as np
import open3d as o3d


def create_mesh_box(size_x, size_y, size_z, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """Create an axis-aligned box mesh using a corner-origin parameterization."""
    mesh = o3d.geometry.TriangleMesh()
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [size_x, 0.0, 0.0],
            [0.0, 0.0, size_z],
            [size_x, 0.0, size_z],
            [0.0, size_y, 0.0],
            [size_x, size_y, 0.0],
            [0.0, size_y, size_z],
            [size_x, size_y, size_z],
        ],
        dtype=np.float64,
    )
    vertices[:, 0] += float(offset_x)
    vertices[:, 1] += float(offset_y)
    vertices[:, 2] += float(offset_z)
    triangles = np.array(
        [
            [4, 7, 5],
            [4, 6, 7],
            [0, 2, 4],
            [2, 6, 4],
            [0, 1, 2],
            [1, 3, 2],
            [1, 5, 7],
            [1, 7, 3],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 1],
            [1, 4, 5],
        ],
        dtype=np.int32,
    )
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


def build_gripper_mesh(
    center,
    rotation_matrix,
    width,
    depth,
    score=1.0,
    color=None,
    height=0.004,
    finger_width=0.004,
    finger_length=0.07,
    tail_length=0.04,
    depth_base=0.02,
):
    """
    Build a debug gripper mesh whose geometry mirrors the collision detector.

    Local frame (same as collision detector):
      X = approach direction (+ toward object)
      Y = width / opening direction
      Z = height direction
      Origin = grasp center (grasp.translation)

    The finger, bottom, and tail boxes are anchored to ``depth`` so that
    their X-ranges match the collision occupancy model:

        finger  : X in (depth - finger_length, depth]
        bottom  : X in (depth - finger_length - finger_width,
                        depth - finger_length]
        tail    : X in (depth - finger_length - finger_width - tail_length,
                        depth - finger_length - finger_width]

    ``depth_base`` is kept for signature compatibility but no longer
    participates in geometry (engine overrides it to 0.0).
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    width = float(width)
    depth = float(depth)
    height = float(height)
    finger_width = float(finger_width)
    finger_length = float(finger_length)
    tail_length = float(tail_length)
    _ = float(depth_base)  # retained for signature compat; not used in geometry

    if color is None:
        color = (float(score), 0.0, 1.0 - float(score))
    color = np.asarray(color, dtype=np.float64).reshape(3)

    # ── left finger ────────────────────────────────────────────
    left = create_mesh_box(
        finger_length, finger_width, height,
        offset_x=depth - finger_length,
        offset_y=-(width / 2.0 + finger_width),
        offset_z=-height / 2.0,
    )

    # ── right finger ───────────────────────────────────────────
    right = create_mesh_box(
        finger_length, finger_width, height,
        offset_x=depth - finger_length,
        offset_y=width / 2.0,
        offset_z=-height / 2.0,
    )

    # ── bottom (full-span base plate, Y matches collision) ────────
    bottom = create_mesh_box(
        finger_width, width + 2.0 * finger_width, height,
        offset_x=depth - finger_length - finger_width,
        offset_y=-(width / 2.0 + finger_width),
        offset_z=-height / 2.0,
    )

    # ── tail (rear extension) ──────────────────────────────────
    tail = create_mesh_box(
        tail_length, finger_width, height,
        offset_x=depth - finger_length - finger_width - tail_length,
        offset_y=-finger_width / 2.0,
        offset_z=-height / 2.0,
    )

    # ── assemble & transform to world frame ────────────────────
    left_verts = np.asarray(left.vertices)
    right_verts = np.asarray(right.vertices)
    bottom_verts = np.asarray(bottom.vertices)
    tail_verts = np.asarray(tail.vertices)

    left_tri = np.asarray(left.triangles)
    right_tri = np.asarray(right.triangles) + 8
    bottom_tri = np.asarray(bottom.triangles) + 16
    tail_tri = np.asarray(tail.triangles) + 24

    vertices = np.concatenate([left_verts, right_verts, bottom_verts, tail_verts], axis=0)
    vertices = np.dot(rotation_matrix, vertices.T).T + center

    triangles = np.concatenate([left_tri, right_tri, bottom_tri, tail_tri], axis=0)
    colors_arr = np.repeat(color[np.newaxis, :], len(vertices), axis=0)

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors_arr)
    return gripper
