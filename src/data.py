"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(
            grid_conf["xbound"], grid_conf["ybound"], grid_conf["zbound"]
        )
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get("sample_data", rec["data"]["CAM_FRONT"])
        imgname = os.path.join(self.nusc.dataroot, sampimg["filename"])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f"{d2}/{d1}/{d0}/{di}/{fi}"

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print("adjusting nuscenes file paths")
            fs = glob(os.path.join(self.nusc.dataroot, "samples/*/samples/CAM*/*.jpg"))
            fs += glob(
                os.path.join(
                    self.nusc.dataroot, "samples/*/samples/LIDAR_TOP/*.pcd.bin"
                )
            )
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f"samples/{di}/{fi}"] = fname
            fs = glob(
                os.path.join(self.nusc.dataroot, "sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin")
            )
            for f in fs:
                di, fi, fname = find_name(f)
                info[f"sweeps/{di}/{fi}"] = fname
            for rec in self.nusc.sample_data:
                if rec["channel"] == "LIDAR_TOP" or (
                    rec["is_key_frame"] and rec["channel"] in self.data_aug_conf["cams"]
                ):
                    rec["filename"] = info[rec["filename"]]

    def get_scenes(self):
        # filter by scene split
        split = {
            "v1.0-trainval": {True: "train", False: "val"},
            "v1.0-mini": {True: "mini_train", False: "mini_val"},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [
            samp
            for samp in samples
            if self.nusc.get("scene", samp["scene_token"])["name"] in self.scenes
        ]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x["scene_token"], x["timestamp"]))

        return samples

    def sample_augmentation(self):
        """
        Randomly (or deterministically) resize and crop the input image to final_dim.

        During training (self.is_train == True):
        1. Randomly select a resize factor in 'resize_lim'.
        2. Compute newW, newH as the resized width and height.
        3. Randomly shift the crop region horizontally within [0, newW - fW].
        4. Vertically, compute 'crop_h' using a random bottom fraction limit so that
            the final crop is fH high, and can shift up/down.
        5. Potentially apply random horizontal flip and random rotation.

        During validation/testing (self.is_train == False):
        1. Compute a deterministic resize factor ensuring the final crop fits.
        2. Center the crop horizontally, and use the mean of 'bot_pct_lim' vertically.

        Returns
        -------
        resize : float
            The chosen resize factor.
        resize_dims : tuple
            The (width, height) after resizing.
        crop : tuple
            The bounding box (left, top, right, bottom) for the final crop.
        flip : bool
            Whether to horizontally flip the image (only in training).
        rotate : float
            The rotation in degrees (only in training).
        """
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]

        if self.is_train:
            # 1) Random resize
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims

            # 2) Random vertical crop
            crop_h = (
                int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            # 3) Random horizontal crop
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

            # 4) Optional flip
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True

            # 5) Random rotation
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])

        else:
            # Validation/test: deterministic crop
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims

            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

            flip = False
            rotate = 0

        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:
            samp = self.nusc.get("sample_data", rec["data"][cam])
            imgname = os.path.join(self.nusc.dataroot, samp["filename"])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
            intrin = torch.Tensor(sens["camera_intrinsic"])
            rot = torch.Tensor(Quaternion(sens["rotation"]).rotation_matrix)
            tran = torch.Tensor(sens["translation"])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (
            torch.stack(imgs),
            torch.stack(rots),
            torch.stack(trans),
            torch.stack(intrins),
            torch.stack(post_rots),
            torch.stack(post_trans),
        )

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec, nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):
        """
        Create a binary BEV mask indicating the footprints of vehicles in the given record.

        1. Retrieves the ego vehicle pose (translation and rotation) to transform all boxes
        into the ego frame.
        2. Initializes a 2D numpy array (img) of shape [nx[0], nx[1]] to zero.
        3. For each annotation (instance) in rec["anns"]:
        - Skips if not a 'vehicle'.
        - Constructs a NuScenes bounding box from translation, size, and rotation.
        - Transforms the box to the ego frame by applying the negative ego translation
            and the inverse ego rotation.
        - Extracts the bottom (x, y) corners, transforms them into the discretized BEV grid,
            and fills the corresponding polygon in img with 1.0.
        4. Returns the final mask as a Torch tensor of shape (1, nx[0], nx[1]).

        Parameters
        ----------
        rec : dict
            A NuScenes record containing 'anns' (annotation tokens) and references to sample data.

        Returns
        -------
        torch.Tensor
            A binary BEV mask with shape (1, nx[0], nx[1]). Cells are 1 where a vehicle is present,
            otherwise 0.
        """
        # 1. Obtain ego pose
        egopose = self.nusc.get(
            "ego_pose",
            self.nusc.get("sample_data", rec["data"]["LIDAR_TOP"])["ego_pose_token"],
        )
        trans = -np.array(egopose["translation"])
        rot = Quaternion(egopose["rotation"]).inverse

        # 2. Initialize binary image
        img = np.zeros((self.nx[0], self.nx[1]))

        # 3. Process annotations
        for tok in rec["anns"]:
            inst = self.nusc.get("sample_annotation", tok)
            if not inst["category_name"].split(".")[0] == "vehicle":
                continue

            # 4. Construct and transform bounding box
            box = Box(inst["translation"], inst["size"], Quaternion(inst["rotation"]))
            box.translate(trans)
            box.rotate(rot)

            # 5. Project corners to BEV
            pts = box.bottom_corners()[:2].T  # shape (4,2)
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.0) / self.dx[:2]
            ).astype(np.int32)

            # swap x,y columns to (row,col)
            pts[:, [1, 0]] = pts[:, [0, 1]]

            # 6. Fill polygon
            cv2.fillPoly(img, [pts], 1.0)

        # 7. Convert to Torch tensor
        return torch.Tensor(img).unsqueeze(0)

    def choose_cams(self):
        if self.is_train and self.data_aug_conf["Ncams"] < len(
            self.data_aug_conf["cams"]
        ):
            cams = np.random.choice(
                self.data_aug_conf["cams"], self.data_aug_conf["Ncams"], replace=False
            )
        else:
            cams = self.data_aug_conf["cams"]
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(
            rec, cams
        )
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec)

        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(
            rec, cams
        )
        binimg = self.get_binimg(rec)

        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(
    version, dataroot, data_aug_conf, grid_conf, bsz, nworkers, parser_name
):
    nusc = NuScenes(
        version="v1.0-{}".format(version),
        dataroot=dataroot,
        verbose=False,
    )
    parser = {
        "vizdata": VizData,
        "segmentationdata": SegmentationData,
    }[parser_name]
    traindata = parser(
        nusc, is_train=True, data_aug_conf=data_aug_conf, grid_conf=grid_conf
    )
    valdata = parser(
        nusc, is_train=False, data_aug_conf=data_aug_conf, grid_conf=grid_conf
    )

    trainloader = torch.utils.data.DataLoader(
        traindata,
        batch_size=bsz,
        shuffle=True,
        num_workers=nworkers,
        drop_last=True,
        worker_init_fn=worker_rnd_init,
    )
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=bsz, shuffle=False, num_workers=nworkers
    )

    return trainloader, valloader
