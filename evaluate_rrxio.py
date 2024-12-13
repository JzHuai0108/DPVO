import glob
import sys
from multiprocessing import Process, Queue

import cv2
import evo.main_ape as main_ape
import numpy as np
import os
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
    images_dir = os.path.join(scene_dir, modality)

    if 'thermal' in modality.lower():
        fx, fy, cx, cy = 334.19639643, 334.26241379, 318.48142004, 250.56663663
        opt_K = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3, 3)
        raw_K = np.array([404.98, 0.0, 319.05, 0.0, 405.06, 251.84, 0.0, 0.0, 1.0]).reshape(3, 3)
        raw_d = np.array([-0.09202092749042277, 0.04012198239889151, -0.03795923559427829, 0.010728597805678742])
        wd, ht = 640, 512
    else:
        fx, fy, cx, cy = 315.93052466, 315.93052466, 318.24482797, 252.56753124
        opt_K = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3, 3)
        raw_K = np.array([372.51, 0.0, 318.77, 0.0, 372.51, 253.24, 0.0, 0.0, 1.0]).reshape(3, 3)
        raw_d = np.array([0.013404032824313381, 0.013570186060580948, -0.011005228038287808, 0.0040591597486])
        wd, ht = 640, 512

    map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
        raw_K, raw_d,
        np.eye(3),
        opt_K,
        (wd, ht),
        cv2.CV_32FC1,
    )

    image_list = sorted(glob.glob(os.path.join(images_dir, "*.png")))[skip::stride]
    for imfile in image_list:
        image = cv2.imread(str(imfile))
        image = cv2.remap(image, map1x, map1y, cv2.INTER_LINEAR)
        image = image.transpose(2,0,1)

        intrinsics = np.asarray([fx, fy, cx, cy])
        queue.put((float(os.path.splitext(os.path.basename(imfile))[0]), image, intrinsics))

    queue.put((-1, image, intrinsics))

@torch.no_grad()
def run(cfg, network, scene_dir, sequence, stride=1, viz=False, show_img=False):
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
    parser.add_argument('--dir', type=str, default="datasets/irs_rtvi_datasets_2021")
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

    scenes = [
        'mocap_easy',
        'gym',
        'mocap_dark',
        'indoor_floor',
        'mocap_medium',
        'mocap_difficult',
        'outdoor_campus',
        'outdoor_street',
        'mocap_dark_fast'
    ]

    modalities = ['visual', 'thermal']

    results = {}
    for modality in modalities:
        os.makedirs(f"trajectory_plots/RRXIO_{modality}", exist_ok=True)
        os.makedirs(f'saved_trajectories/RRXIO_{modality}', exist_ok=True)
        for scene in scenes:
            scene_dir = os.path.join(args.dir, scene)
            groundtruth = os.path.join(scene_dir, f"gt_{modality}.txt")
            traj_ref = file_interface.read_tum_trajectory_file(groundtruth)

            scene_results = []
            for trial_num in range(args.trials):
                traj_est, timestamps = run(cfg, args.network, scene_dir, scene, args.stride, args.viz, args.show_img)

                traj_est = PoseTrajectory3D(
                    positions_xyz=traj_est[:,:3],
                    orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
                    timestamps=timestamps)

                if args.save_trajectory:
                    os.makedirs(f'saved_trajectories/RRXIO_{modality}/laptop_RRXIO_{modality}_{scene}', exist_ok=True)
                    if trial_num == 0:
                        file_interface.write_tum_trajectory_file(f"saved_trajectories/RRXIO_{modality}/laptop_RRXIO_{modality}_{scene}/"
                                                             f"stamped_groundtruth.txt", traj_ref)
                    file_interface.write_tum_trajectory_file(f"saved_trajectories/RRXIO_{modality}/laptop_RRXIO_{modality}_{scene}/"
                                                             f"stamped_traj_estimate{trial_num:01d}.txt", traj_est)

                traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

                result = main_ape.ape(traj_ref, traj_est, est_name='traj',
                    pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
                ate_score = result.stats["rmse"]

                if args.plot:
                    plot_trajectory(traj_est, traj_ref, f"RRXIO {modality} {scene} Trial #{trial_num+1} (ATE: {ate_score:.03f})",
                                    f"trajectory_plots/RRXIO_{modality}/{scene}_Trial{trial_num+1:02d}.pdf", align=True, correct_scale=True)


                scene_results.append(ate_score)

            results[scene] = np.median(scene_results)
            print(modality, scene, sorted(scene_results))

        xs = []
        for scene in results:
            print(modality, scene, results[scene])
            xs.append(results[scene])

        print(f"{modality} AVG: {np.mean(xs)}")
