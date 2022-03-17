# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import skimage.transform
import numpy as np
import PIL.Image as pil
import math

from manydepth.kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset
from manydepth.utils import euler2mat


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        return folder, frame_index, side

    def check_GpsPose_T(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        gps_filename = os.path.join(
            self.data_path,
            scene_name,
            "oxts/data/{:010d}.txt".format(int(frame_index)))

        return os.path.isfile(gps_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    # def get_pose_T(self,folder, frame_index):
    #
    #     def Calculate_T(folder, frame_index):
    #         re = 6378137.0
    #
    #         oxts_start = os.path.join(self.data_path, folder, "oxts/data", "0000000000.txt")
    #         oxts_index = os.path.join(self.data_path, folder, "oxts/data", "{:010d}.txt".format(frame_index))
    #
    #         gps_start = np.genfromtxt(oxts_start).astype(np.float32)
    #         gps_index = np.genfromtxt(oxts_index).astype(np.float32)
    #
    #         angle = [gps_index[3], gps_index[4], gps_index[5]]
    #         rotMat = euler2mat(angle)
    #
    #         x_index = math.cos((math.pi * gps_start[0]) / 180.0) * math.pi * gps_index[1] * re / 180.0
    #         y_index = math.cos((math.pi * gps_start[0]) / 180.0) * math.log(math.tan((math.pi * (90 + gps_index[0])) / 360.0)) * re
    #         z_index = gps_index[2]
    #
    #         translation = np.array([x_index, y_index, z_index]).astype(np.float32)
    #         dummy = np.array([0, 0, 0, 1]).astype(np.float32)
    #
    #         T = np.column_stack((rotMat, translation))
    #         T = np.row_stack((T, dummy))
    #
    #         return T
    #
    #     initial_T = Calculate_T(folder, 0)
    #     inv_initial_T = np.linalg.inv(initial_T)
    #
    #     T_now = Calculate_T(folder, frame_index)
    #     T_now = np.matmul(inv_initial_T, T_now)
    #
    #     return np.array([T_now[0][3], T_now[1][3], T_now[2][3]]).astype(np.float32)

    def get_pose_T(self,folder, frame_index):
        re=6378137.0
        oxts_start=os.path.join(self.data_path,folder,"oxts/data","0000000000.txt")
        oxts_index=os.path.join(self.data_path,folder,"oxts/data","{:010d}.txt".format(frame_index))

        gps_start=np.genfromtxt(oxts_start).astype(np.float32)
        gps_index=np.genfromtxt(oxts_index).astype(np.float32)

        x_start = math.cos((math.pi * gps_start[0]) / 180.0) * math.log(math.tan((math.pi * (90 + gps_index[0])) / 360.0)) * re
        y_start = gps_index[2]
        z_start = math.cos((math.pi * gps_start[0]) / 180.0) * math.pi * gps_index[1] *re / 180.0

        return np.array([x_start, y_start, z_start]).astype(np.float32)



class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
