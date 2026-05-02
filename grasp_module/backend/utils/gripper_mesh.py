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
    tail_length=0.04,
    depth_base=0.02,
):
    """
    Build a debug gripper mesh in repo-local code so runtime behavior does not
    depend on graspnetAPI's installed signature.
    """
    center = np.asarray(center, dtype=np.float64).reshape(3)
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    width = float(width)
    depth = float(depth)
    height = float(height)
    finger_width = float(finger_width)
    tail_length = float(tail_length)
    depth_base = float(depth_base)

    if color is None:
        color = (float(score), 0.0, 1.0 - float(score))
    color = np.asarray(color, dtype=np.float64).reshape(3)

    left = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    right = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_vertices = np.asarray(left.vertices).copy()
    left_triangles = np.asarray(left.triangles).copy()
    left_vertices[:, 0] -= depth_base + finger_width
    left_vertices[:, 1] -= width / 2.0 + finger_width
    left_vertices[:, 2] -= height / 2.0

    right_vertices = np.asarray(right.vertices).copy()
    right_triangles = np.asarray(right.triangles).copy() + 8
    right_vertices[:, 0] -= depth_base + finger_width
    right_vertices[:, 1] += width / 2.0
    right_vertices[:, 2] -= height / 2.0

    bottom_vertices = np.asarray(bottom.vertices).copy()
    bottom_triangles = np.asarray(bottom.triangles).copy() + 16
    bottom_vertices[:, 0] -= finger_width + depth_base
    bottom_vertices[:, 1] -= width / 2.0
    bottom_vertices[:, 2] -= height / 2.0

    tail_vertices = np.asarray(tail.vertices).copy()
    tail_triangles = np.asarray(tail.triangles).copy() + 24
    tail_vertices[:, 0] -= tail_length + finger_width + depth_base
    tail_vertices[:, 1] -= finger_width / 2.0
    tail_vertices[:, 2] -= height / 2.0

    vertices = np.concatenate(
        [left_vertices, right_vertices, bottom_vertices, tail_vertices],
        axis=0,
    )
    vertices = np.dot(rotation_matrix, vertices.T).T + center
    triangles = np.concatenate(
        [left_triangles, right_triangles, bottom_triangles, tail_triangles],
        axis=0,
    )
    colors = np.repeat(color[np.newaxis, :], len(vertices), axis=0)

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper
