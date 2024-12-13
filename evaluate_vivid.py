import glob
import os
import sys
from multiprocessing import Process, Queue

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface, plot

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory
from dpvo.utils import Timer

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

def image_stream(queue, scene_dir, modality, sequence, stride, skip=0):
    """ image generator """
    images_dir = os.path.join(scene_dir, modality, 'data')
    if 'thermal' in modality.lower():
        fx, fy, cx, cy = 437.38861083256637, 437.29475745770907, 323.5284494924228, 256.36315482047905
    else: # RGB
        fx, fy, cx, cy = 531.0787710720505, 530.9280349280389, 322.0434384516226, 239.58555406488455

    image_list = sorted(glob.glob(os.path.join(images_dir, "*.png")))[skip::stride]
    for i, imfile in enumerate(image_list):
        image = cv2.imread(imfile)
        image = image.transpose(2,0,1)
        intrinsics = np.asarray([fx, fy, cx, cy])
        queue.put((float(os.path.splitext(os.path.basename(imfile))[0]), image, intrinsics))

    queue.put((-1, image, intrinsics))

@torch.no_grad()
def run(cfg, network, scene_dir, modality, sequence, stride=1, viz=False, show_img=False):

    slam = None

    queue = Queue(maxsize=8)
    reader = Process(target=image_stream, args=(queue, scene_dir, modality, sequence, stride, 0))
    reader.start()

    for step in range(sys.maxsize):
        (t, images, intrinsics) = queue.get()
        if t < 0: break

        images = torch.as_tensor(images, device='cuda')
        intrinsics = torch.as_tensor(intrinsics, dtype=torch.float, device='cuda')

        if show_img:
            show_image(images[0], 1)

        if slam is None:
            slam = DPVO(cfg, network, ht=images.shape[-2], wd=images.shape[-1], viz=viz)

        intrinsics = intrinsics.cuda()

        with Timer("SLAM", enabled=False):
            slam(t, images, intrinsics)

    reader.join()

    poses, tstamps = slam.terminate()
    np.save(f"poses_{sequence}.npy", poses)
    np.save(f"tstamps_{sequence}.npy", tstamps)
    return poses, tstamps


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--dir', type=str, default="datasets/VIVID")
    parser.add_argument('--backend_thresh', type=float, default=64.0)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()
    args.plot = True
    args.save_trajectory = True

    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)

    print("\nRunning with config...")
    print(cfg, "\n")

    torch.manual_seed(1234)

    modalities = ['RGB', 'Thermal_fs', 'Thermal_vis', 'Thermal_naive', 'Thermal_shin']
    scenes = ['outdoor_robust_day1', 'outdoor_robust_day2', 'outdoor_robust_night1', 'outdoor_robust_night2',
              'indoor_aggresive_dark', 'indoor_aggresive_global', 'indoor_aggresive_local', 'indoor_robust_dark',
              'indoor_robust_global', 'indoor_robust_local', 'indoor_robust_varying', 'indoor_unstable_dark',
              'indoor_unstable_global', 'indoor_unstable_local']

    results = {}
    for modality in modalities:
        os.makedirs(f"trajectory_plots/VIVID_{modality}", exist_ok=True)
        os.makedirs(f'saved_trajectories/VIVID_{modality}', exist_ok=True)
        for scene in scenes:
            scene_dir = os.path.join(args.dir, scene)
            postfix = 'RGB'
            if 'thermal' in modality.lower():
                postfix = 'thermal'
            images_dir = os.path.join(scene_dir, modality, 'data')
            image_list = sorted(glob.glob(os.path.join(images_dir, "*.png")))
            groundtruth = os.path.join(scene_dir, f"poses_{postfix}.txt")
            poses_ref = file_interface.read_kitti_poses_file(groundtruth)
            num_imgs = len(image_list)
            assert num_imgs <= poses_ref.num_poses
            gt_times = np.arange(1, num_imgs + 1, dtype=np.float64)
            traj_ref = PoseTrajectory3D(
                positions_xyz=poses_ref.positions_xyz[:num_imgs, :],
                orientations_quat_wxyz=poses_ref.orientations_quat_wxyz[:num_imgs, :],
                timestamps=gt_times)

            scene_results = []
            for trial_num in range(args.trials):
                traj_est, timestamps = run(cfg, args.network, scene_dir, modality, scene, args.stride, args.viz, args.show_img)

                traj_est = PoseTrajectory3D(
                    positions_xyz=traj_est[:,:3],
                    orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
                    timestamps=timestamps)

                print(f'dpvo timestamps {timestamps[:5]} len {len(timestamps)}, gt times {gt_times[:5]} len {len(gt_times)}')
                assert len(timestamps) == len(gt_times)
                if args.save_trajectory:
                    os.makedirs(f'saved_trajectories/VIVID_{modality}/laptop_VIVID_{modality}_{scene}', exist_ok=True)
                    if trial_num == 0:
                        file_interface.write_tum_trajectory_file(f"saved_trajectories/VIVID_{modality}/laptop_VIVID_{modality}_{scene}/"
                                                             f"stamped_groundtruth.txt", traj_ref)
                    file_interface.write_tum_trajectory_file(f"saved_trajectories/VIVID_{modality}/laptop_VIVID_{modality}_{scene}/"
                                                             f"stamped_traj_estimate{trial_num:01d}.txt", traj_est)

                traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

                result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                    pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
                ate_score = result.stats["rmse"]

                if args.plot:
                    plot_trajectory(traj_est, traj_ref, f"VIVID {modality} {scene} Trial #{trial_num+1} (ATE: {ate_score:.03f})",
                                    f"trajectory_plots/VIVID_{modality}/{scene}_Trial{trial_num+1:02d}.pdf", align=True, correct_scale=True)

                scene_results.append(ate_score)

            results[scene] = np.median(scene_results)
            print(modality, scene, sorted(scene_results))

        xs = []
        for scene in results:
            print(modality, scene, results[scene])
            xs.append(results[scene])
        print(f"{modality} AVG: {np.mean(xs)}")
