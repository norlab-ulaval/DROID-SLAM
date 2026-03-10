import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend
from trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process


class Droid:
    def __init__(self, args):
        super(Droid, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.disable_vis = args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from visualizer.droid_visualizer import visualization_fn
            self.visualizer = Process(target=visualization_fn, args=(self.video, None))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)


    def load_weights(self, weights):
        """ load trained model weights """

        print(weights)
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def track(self, tstamp, image, depth=None, intrinsics=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(tstamp, image, depth, intrinsics)

            # local bundle adjustment
            self.frontend()

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        print("Starting Global Bundle Adjustment...")
        torch.cuda.empty_cache()
        print("Running Global BA (7 steps)...")
        self.backend(7)

        torch.cuda.empty_cache()
        print("Running Global BA (12 steps)...")
        self.backend(12)
        print("Global Bundle Adjustment complete.")

        if hasattr(self.args, 'weight_output_dir') and self.args.weight_output_dir:
            import os
            import cv2
            import numpy as np
            print("Saving confidence maps...")
            graph = self.backend.graph
            if graph.weight.shape[1] > 0:
                weights = graph.weight.cpu().numpy()
                ii = graph.ii.cpu().numpy()
                jj = graph.jj.cpu().numpy()
                tstamps = self.video.tstamp.cpu().numpy()
                if not os.path.exists(self.args.weight_output_dir):
                    os.makedirs(self.args.weight_output_dir)
                for k in range(weights.shape[1]):
                    w = weights[0, k]
                    w_mean = np.mean(w, axis=-1)
                    w_img = (w_mean * 255).astype(np.uint8)
                    t1 = tstamps[ii[k]]
                    t2 = tstamps[jj[k]]
                    fname = os.path.join(self.args.weight_output_dir, f"{t1:.6f}_{t2:.6f}.png")
                    cv2.imwrite(fname, w_img)
                print("Confidence maps saved.")
            else:
                print("No weights found in graph.")

        # camera_trajectory = self.traj_filler(stream)
        # return camera_trajectory.inv().data.cpu().numpy()

