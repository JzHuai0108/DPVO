import os
from os import times

import cv2
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path
from itertools import chain

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
import yaml


def image_msg_to_cv2(image_msg):
    """Convert a sensor_msgs/Image message to an OpenCV image (NumPy array)."""
    # Convert the image data to a NumPy array
    dtype = np.uint8 if image_msg.encoding == 'mono8' else np.uint16 if image_msg.encoding == 'mono16' else np.uint8
    image_np = np.frombuffer(image_msg.data, dtype=dtype).reshape(image_msg.height, image_msg.width, -1)

    # Handle different encodings
    if image_msg.encoding == 'rgb8':
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif image_msg.encoding == 'bgr8':
        return image_np
    elif image_msg.encoding == 'mono8' or image_msg.encoding == 'mono16':
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    return image_np


def image_stream(queue, imagedir, calib, stride, skip=0):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    image_list = sorted(chain.from_iterable(Path(imagedir).glob(e) for e in img_exts))[skip::stride]
    assert os.path.exists(imagedir), imagedir

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        if 0:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)
            intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])

        else:
            intrinsics = np.array([fx, fy, cx, cy])
            
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        queue.put((t, image, intrinsics))

    queue.put((-1, image, intrinsics))


def bag_stream(queue, bagfile, calib_yaml, stride, skip=0):
    image_list = []
    with open(calib_yaml, 'r') as file:
        config = yaml.safe_load(file)
    raw_calib = config['Dataset']['Calibration']['raw']
    img_topic = config['Dataset']['img_topic']
    raw_K = np.eye(3)
    raw_K[0,0] = raw_calib['fx']
    raw_K[0,2] = raw_calib['cx']
    raw_K[1,1] = raw_calib['fy']
    raw_K[1,2] = raw_calib['cy']
    width = raw_calib['width']
    height = raw_calib['height']

    opt_calib = config['Dataset']['Calibration']['opt']
    opt_K = np.eye(3)
    opt_K[0,0] = opt_calib['fx']
    opt_K[0,2] = opt_calib['cx']
    opt_K[1,1] = opt_calib['fy']
    opt_K[1,2] = opt_calib['cy']

    if 'distortion_model' in raw_calib.keys():
        distortion_model = raw_calib['distortion_model']
    else:
        distortion_model = None
    print(f"Distortion model: {distortion_model}")

    if distortion_model == 'radtan':
        dist_coeffs = np.array(
            [
                raw_calib["k1"],
                raw_calib["k2"],
                raw_calib["p1"],
                raw_calib["p2"],
                raw_calib["k3"],
            ]
        )
        map1x, map1y = cv2.initUndistortRectifyMap(
            raw_K,
            dist_coeffs,
            np.eye(3),
            opt_K,
            (width, height),
            cv2.CV_32FC1,
        )
    elif distortion_model == 'equidistant':
        dist_coeffs = np.array(
            [
                raw_calib["k1"],
                raw_calib["k2"],
                raw_calib["k3"],
                raw_calib["k4"]
            ]
        )
        map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
            raw_K,
            dist_coeffs,
            np.eye(3),
            opt_K,
            (width, height),
            cv2.CV_32FC1,
        )
    else:
        map1x, map1y = None, None

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bagfile) as reader:
        # Topic and msgtype information is available on .connections list.
        # for connection in reader.connections:
        #     print(connection.topic, connection.msgtype)
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == img_topic:
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                rgb_image = image_msg_to_cv2(msg)
                # Display the image (optional)
                cv2.imshow('Image', rgb_image)
                cv2.waitKey(1)  # Adjust the delay as needed (e.g., for video playback)

                seqid = msg.header.seq
                time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if distortion_model is None:
                    undistorted_image = rgb_image
                else:
                    undistorted_image = cv2.remap(rgb_image, map1x, map1y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                cv2.imshow('Undistorted', undistorted_image)
                cv2.waitKey(1)  # Adjust the delay as needed (e.g., for video playback)

                image_list.append(undistorted_image)

    for t, image in enumerate(image_list[skip::stride]):
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16] # remove the last rows and columns of the image do not affect the K matrix.
        intrinsics = np.array([opt_calib['fx'], opt_calib['fy'], opt_calib['cx'], opt_calib['cy']])
        queue.put((t, image, intrinsics))

    queue.put((-1, image, intrinsics))


def video_stream(queue, imagedir, calib, stride, skip=0):
    """ video generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    assert os.path.exists(imagedir), imagedir
    cap = cv2.VideoCapture(imagedir)

    t = 0

    for _ in range(skip):
        ret, image = cap.read()

    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            ret, image = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

        if not ret:
            break

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        intrinsics = np.array([fx*.5, fy*.5, cx*.5, cy*.5])
        queue.put((t, image, intrinsics))

        t += 1

    queue.put((-1, image, intrinsics))
    cap.release()

