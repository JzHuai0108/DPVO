import numpy as np
from evo.core.trajectory import PoseTrajectory3D


def write_tum_trajectory_file(file_path: str, traj: PoseTrajectory3D,
                              confirm_overwrite: bool = False) -> None:
    """
    :param file_path: desired text file for trajectory (string or handle)
    :param traj: trajectory.PoseTrajectory3D
    :param confirm_overwrite: whether to require user interaction
           to overwrite existing files
    """
    if not isinstance(traj, PoseTrajectory3D):
        raise RuntimeError(
            "trajectory must be a PoseTrajectory3D object")
    stamps = traj.timestamps
    xyz = traj.positions_xyz
    # shift -1 column -> w in back column
    quat = np.roll(traj.orientations_quat_wxyz, -1, axis=1)
    mat = np.column_stack((stamps, xyz, quat))
    np.savetxt(file_path, mat, delimiter=" ", fmt="%.9f %.6f %.6f %.6f %.9f %.9f %.9f %.9f")
    if isinstance(file_path, str):
        print("Trajectory saved to: " + file_path)
