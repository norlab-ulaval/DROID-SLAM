import queue
import sys
import threading

from tqdm import tqdm

sys.path.append("droid_slam")

import argparse
import concurrent.futures
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from droid import Droid
from droid_async import DroidAsync
from lietorch import SE3


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow("image", image / 255.0)
    cv2.waitKey(1)


def image_stream(imagedir, calib, stride, stereo, image_size, timestamps_filepath=None):
    """image generator with batched multithreaded prefetching"""

    K_l = K_r = None
    D_l = D_r = None

    if stereo:
        calib_l_path = os.path.join(imagedir, "calib", "zedx_left.json")
        calib_r_path = os.path.join(imagedir, "calib", "zedx_right.json")

        if not os.path.exists(calib_l_path) or not os.path.exists(calib_r_path):
            print(
                f"Error: Stereo calibration files not found at {calib_l_path} or {calib_r_path}"
            )
            sys.exit(1)

        with open(calib_l_path, "r") as f:
            data_l = json.load(f)
            k_l = data_l["k"]
            K_l = np.array(k_l).reshape(3, 3)
            D_l = np.array(data_l["d"])

        with open(calib_r_path, "r") as f:
            data_r = json.load(f)
            k_r = data_r["k"]
            K_r = np.array(k_r).reshape(3, 3)
            D_r = np.array(data_r["d"])

        fx, fy, cx, cy = K_l[0, 0], K_l[1, 1], K_l[0, 2], K_l[1, 2]

    else:
        calib = np.loadtxt(calib, delimiter=" ")
        fx, fy, cx, cy = calib[:4]

        K = np.eye(3)
        K[0, 0] = fx
        K[0, 2] = cx
        K[1, 1] = fy
        K[1, 2] = cy
        K_l = K

    # Load allowed timestamps from file (first column, in seconds -> convert to microseconds)
    allowed_timestamps_us = None
    if timestamps_filepath is not None:
        allowed_timestamps_us = set()
        with open(timestamps_filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ts_s = float(line.split()[0])
                allowed_timestamps_us.add(round(ts_s * 1e6))

    def filter_by_timestamps(image_list):
        if allowed_timestamps_us is None:
            return image_list
        return [
            fname for fname in image_list
            if round(float(fname.split(".")[0])) in allowed_timestamps_us
        ]

    if stereo:
        imagedir_left = os.path.join(imagedir, "zedx_left")
        imagedir_right = os.path.join(imagedir, "zedx_right")
        image_list_left = filter_by_timestamps(sorted(os.listdir(imagedir_left))[::stride])
        image_list_right = filter_by_timestamps(sorted(os.listdir(imagedir_right))[::stride])
        assert len(image_list_left) == len(image_list_right)
    else:
        image_list = filter_by_timestamps(sorted(os.listdir(imagedir))[::stride])

    total_images = len(image_list_left if stereo else image_list)

    # Isolated worker function for a single frame index
    def process_frame(t):
        imfile = image_list_left[t] if stereo else image_list[t]

        if stereo:
            image_left = cv2.imread(os.path.join(imagedir_left, image_list_left[t]))
            image_right = cv2.imread(os.path.join(imagedir_right, image_list_right[t]))

            image_left = cv2.undistort(image_left, K_l, D_l)
            image_right = cv2.undistort(image_right, K_r, D_r)
            images = [image_left, image_right]
        else:
            image = cv2.imread(os.path.join(imagedir, imfile))
            if len(calib) > 4:
                image = cv2.undistort(image, K_l, calib[4:])
            images = [image]

        h0, w0, _ = images[0].shape
        h1 = image_size[0]
        w1 = image_size[1]

        images = [cv2.resize(img, (w1, h1)) for img in images]
        images = [img[: h1 - h1 % 8, : w1 - w1 % 8] for img in images]
        images = [torch.as_tensor(img).permute(2, 0, 1) for img in images]

        images = torch.stack(images)

        baseline = 0.1
        intrinsics = torch.as_tensor([fx, fy, cx, cy, baseline])
        intrinsics[0:4:2] *= w1 / w0
        intrinsics[1:4:2] *= h1 / h0

        tstamp_s = float(imfile.split(".")[0]) / 1e6

        return t, tstamp_s, images, intrinsics

    # The background thread that manages the batches
    def batch_fetcher(batch_queue, batch_size, num_workers):
        for batch_start in range(0, total_images, batch_size):
            batch_end = min(batch_start + batch_size, total_images)
            indices_to_process = range(batch_start, batch_end)

            batch_results = []

            # Use threads to fetch the chunk concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_idx = {executor.submit(process_frame, idx): idx for idx in indices_to_process}

                for future in concurrent.futures.as_completed(future_to_idx):
                    try:
                        batch_results.append(future.result())
                    except Exception as exc:
                        idx = future_to_idx[future]
                        print(f"Frame {idx} generated an exception: {exc}")

            # Sort the completed batch chronologically
            batch_results.sort(key=lambda x: x[0])

            # Put the sorted batch in the queue.
            # If the queue is full (maxsize=1), this blocks until SLAM finishes the current batch.
            batch_queue.put(batch_results)

        # Sentinel value to signal the generator that all batches are done
        batch_queue.put(None)

    # The main generator function yielded to SLAM
    def generator():
        batch_size = 300
        num_workers = 20

        # maxsize=1 means memory holds maximum 2 batches at a time
        # (1 processing in SLAM, 1 waiting in queue)
        batch_queue = queue.Queue(maxsize=1)

        # Start the background fetcher
        fetcher_thread = threading.Thread(
            target=batch_fetcher,
            args=(batch_queue, batch_size, num_workers),
            daemon=True
        )
        fetcher_thread.start()

        # Continually pull ready batches from the queue
        while True:
            current_batch = batch_queue.get()

            # If we hit the sentinel value, break out
            if current_batch is None:
                break

            # Yield frames to the SLAM main loop
            for res in current_batch:
                _, tstamp_s, images, intrinsics = res
                yield tstamp_s, images, intrinsics

    return generator(), total_images


def save_reconstruction(droid, save_path):

    if hasattr(droid, "video2"):
        video = droid.video2
    else:
        video = droid.video

    t = video.counter.value
    print(f"Extracting reconstruction data for {t} frames...")
    save_data = {
        "tstamps": video.tstamp[:t].cpu(),
        "images": video.images[:t].cpu(),
        "disps": video.disps_up[:t].cpu(),
        "poses": video.poses[:t].cpu(),
        "intrinsics": video.intrinsics[:t].cpu(),
    }

    print(f"Saving reconstruction map to {save_path}...")
    torch.save(save_data, save_path)
    print("Reconstruction saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--timestamps_path", type=str, help="path to timestamps file")

    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="weight for translation / rotation components of flow",
    )
    parser.add_argument(
        "--filter_thresh",
        type=float,
        default=2.4,
        help="how much motion before considering new keyframe",
    )
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument(
        "--keyframe_thresh",
        type=float,
        default=4.0,
        help="threshold to create a new keyframe",
    )
    parser.add_argument(
        "--frontend_thresh",
        type=float,
        default=16.0,
        help="add edges between frames whithin this distance",
    )
    parser.add_argument(
        "--frontend_window", type=int, default=25, help="frontend optimization window"
    )
    parser.add_argument(
        "--frontend_radius",
        type=int,
        default=2,
        help="force edges between frames within radius",
    )
    parser.add_argument(
        "--frontend_nms", type=int, default=1, help="non-maximal supression of edges"
    )

    parser.add_argument("--pgo", action="store_true")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")

    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    parser.add_argument(
        "--trajectory_path", type=str, help="path to save trajectory file"
    )
    parser.add_argument(
        "--weight_output_dir",
        type=str,
        help="path to directory to save confidence maps",
    )
    parser.add_argument("--stereo", action="store_true")
    parser.add_argument(
        "--max_frames", type=int, default=-1, help="max frames to evaluate"
    )
    args = parser.parse_args()

    if args.reconstruction_path is not None and not args.reconstruction_path.endswith(
        (".pt", ".pth")
    ):
        args.reconstruction_path = os.path.join(
            args.reconstruction_path, "reconstruction.pth"
        )

    if args.weight_output_dir is None:
        if args.reconstruction_path:
            args.weight_output_dir = os.path.join(
                os.path.dirname(args.reconstruction_path), "exported_weights"
            )
        elif args.trajectory_path:
            args.weight_output_dir = os.path.join(
                os.path.dirname(args.trajectory_path), "exported_weights"
            )
        else:
            args.weight_output_dir = "exported_weights"

    torch.multiprocessing.set_start_method("spawn")

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    # Setup incremental odometry file
    if args.trajectory_path:
        incremental_odom_path = os.path.splitext(args.trajectory_path)[0] + "_odom.txt"
        output_folder = os.path.dirname(args.trajectory_path)
    else:
        output_folder = (
            os.path.dirname(args.reconstruction_path)
            if args.reconstruction_path
            else "."
        )
        incremental_odom_path = os.path.join(output_folder, "trajectory_odom.txt")

    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open in write mode once to clear/create the file
    open(incremental_odom_path, "w").close()

    start_time = time.time()
    image_gen, num_of_images = image_stream(
        args.imagedir, args.calib, args.stride, args.stereo, args.image_size, args.timestamps_path
    )
    for t, (tstamp, image, intrinsics) in enumerate(image_gen):
        frame_time = time.time()
        if args.max_frames > 0 and t >= args.max_frames:
            break
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            image_size = [image.shape[2], image.shape[3]]
            droid = DroidAsync(args) if args.asynchronous else Droid(args)
            print(f"Initialized DROID-SLAM tracking. Frame size: {image_size}")

        droid.track(tstamp, image, intrinsics=intrinsics)

        odom_video = getattr(droid, "video1", getattr(droid, "video", None))

        if odom_video is not None and odom_video.counter.value > 0:
            idx = odom_video.counter.value - 1
            latest_tstamp = odom_video.tstamp[idx].cpu().item()

            # Slice [idx:idx+1] to preserve the 2D tensor shape needed by SE3
            latest_pose = odom_video.poses[idx : idx + 1]

            # Convert to Camera-to-World
            latest_pose_c2w = SE3(latest_pose).inv().data[0].cpu().numpy()

            with open(incremental_odom_path, "a") as f:
                p = latest_pose_c2w
                f.write(
                    f"{latest_tstamp} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]}\n"
                )

        elapsed = time.time() - start_time
        fps = (t + 1) / elapsed if elapsed > 0 else 0
        print(
            f"Processed frame {t}/{num_of_images} (tstamp={tstamp:.2f}) in {time.time() - frame_time:.2f} seconds."
        )

    print(
        f"Finished tracking {t + 1} frames. Total time: {time.time() - start_time:.2f} seconds."
    )

    # Save Pre-SLAM Odometry Trajectory
    if hasattr(droid, "video1"):
        odom_video = droid.video1
    elif hasattr(droid, "video"):
        odom_video = droid.video
    else:
        odom_video = None

    if args.pgo:
        print("Terminating tracking and extracting poses...")
        image_gen, _ = image_stream(args.imagedir, args.calib, args.stride, args.stereo, args.image_size)
        traj_est = droid.terminate(image_gen)

        # Save Trajectory (TUM Format) using C2W poses
        if hasattr(droid, "video2"):
            video = droid.video2
        else:
            video = droid.video

        tstamps = video.tstamp[: video.counter.value].cpu().numpy()
        poses_c2w = SE3(video.poses[: video.counter.value]).inv().data.cpu().numpy()

        if args.trajectory_path:
            traj_path = args.trajectory_path
            output_folder = os.path.dirname(traj_path)
        else:
            output_folder = (
                os.path.dirname(args.reconstruction_path)
                if args.reconstruction_path
                else "."
            )
            traj_path = os.path.join(output_folder, "trajectory.txt")

        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print(f"Saving trajectory to {traj_path}...")
        with open(traj_path, "w") as f:
            for i in range(len(poses_c2w)):
                p = poses_c2w[i]
                timestamp = tstamps[i]
                f.write(
                    f"{timestamp} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]}\n"
                )

        if args.reconstruction_path is not None:
            save_reconstruction(droid, args.reconstruction_path)
