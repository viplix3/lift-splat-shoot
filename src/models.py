"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320 + 112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, : self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self.trunk._blocks
                )  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints["reduction_{}".format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints["reduction_{}".format(len(endpoints) + 1)] = x
        x = self.up1(endpoints["reduction_5"], endpoints["reduction_4"])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(
            self.grid_conf["xbound"],
            self.grid_conf["ybound"],
            self.grid_conf["zbound"],
        )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self):
        """
        Create a 3D frustum grid for the downsampled image plane plus depth dimension.

        The frustum defines a set of 3D points in (x, y, d) space:
        - (x, y) covers the pixel locations in the feature map (downsampled image),
        - d covers a sequence of depth values from d_min to d_max in dbound.

        Returns
        -------
        nn.Parameter
            A tensor of shape (D, fH, fW, 3), where:
                D is the number of depth bins,
                fH and fW are the downsampled height and width,
            and each element is (x_pixel, y_pixel, depth).
            The parameter is set to not require gradient.
        """
        # Extract final image dimensions
        ogfH, ogfW = self.data_aug_conf["final_dim"]

        # Downsampled feature map size
        fH, fW = ogfH // self.downsample, ogfW // self.downsample

        # Depth bins
        ds = (
            torch.arange(*self.grid_conf["dbound"], dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        # x-coordinates in [0..(ogfW-1)], downsampled
        xs = (
            torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        # y-coordinates in [0..(ogfH-1)], downsampled
        ys = (
            torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        # Stack into (D, fH, fW, 3) for (x, y, d)
        frustum = torch.stack((xs, ys, ds), dim=-1)

        # Store as a parameter (no gradients needed)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """
        Compute 3D points in the ego vehicle frame from the frustum coordinates.

        This function:
        1) Subtracts and inverts the post-translations and post-rotations to
            'undo' the data augmentations applied to the images.
        2) Converts the (x, y, depth) points from image space to 3D camera space
            by multiplying (x, y) with depth (z).
        3) Applies camera intrinsics (inverse) and extrinsics (rots, trans) to map
            those camera coordinates into the ego vehicle frame.

        Parameters
        ----------
        rots : torch.Tensor (B, N, 3, 3)
            Rotation matrices for each camera from camera-to-ego frame.
        trans : torch.Tensor (B, N, 3)
            Translation vectors for each camera from camera-to-ego frame.
        intrins : torch.Tensor (B, N, 3, 3)
            Camera intrinsic matrices.
        post_rots : torch.Tensor (B, N, 3, 3)
            Rotation matrices of the augmentation transforms applied to the images.
        post_trans : torch.Tensor (B, N, 3)
            Translation vectors of the augmentation transforms applied to the images.

        Returns
        -------
        points : torch.Tensor (B, N, D, H, W, 3)
            The 3D points in ego coordinates for each depth bin, pixel location,
            camera, and batch element.

        Notes
        -----
        - self.frustum is expected to have shape (D, H, W, 3), and is broadcasted
        to (B, N, D, H, W, 3).
        - The transformations are applied in reverse order of how the augmentations
        were generated, then we do camera-to-ego transformations.
        """

        B, N, _ = trans.shape

        # 1) Undo the post-transformation in image space
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )

        # 2) Convert (x, y, depth) -> (X_cam, Y_cam, Z_cam)
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            dim=5,
        )

        # 3) Apply camera intrinsics inverse and extrinsics
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)

        # 4) Add camera-to-ego translation
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """
        Encode camera features, reshaping to/from (B, N) format and returning a
        6D tensor with a specific layout.

        The input 'x' has shape:
            (B, N, C, imH, imW)
        where B = batch size, N = number of cameras, C = input channels,
        imH and imW are the feature map dimensions.

        Steps:
        1) Flatten B and N into one dimension for camera encoding: (B*N, C, imH, imW).
        2) Pass through camera encoder self.camencode(...).
        3) Reshape to (B, N, camC, D, imH//downsample, imW//downsample),
            where camC and D depend on the encoder's output channels.
        4) Permute dimensions to (B, N, D, imH//downsample, imW//downsample, camC).

        Returns
        -------
        torch.Tensor
            A feature tensor of shape (B, N, D, H/downsample, W/downsample, C).
        """
        B, N, C, imH, imW = x.shape

        # (B*N, C, imH, imW)
        x = x.view(B * N, C, imH, imW)

        # Camera encoding (e.g., a CNN)
        x = self.camencode(x)

        # (B, N, camC, D, imH//downsample, imW//downsample)
        x = x.view(
            B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample
        )

        # (B, N, D, imH//downsample, imW//downsample, camC)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        """
        Voxel-pool the per-camera, per-pixel features into a BEV volume, then collapse Z.

        Parameters
        ----------
        geom_feats : torch.Tensor
            Shape (B, N, D, H, W, 3). For each batch, camera, depth-bin, height, width,
            this contains 3D coordinates (X, Y, Z) in some continuous or integer space.
        x : torch.Tensor
            Shape (B, N, D, H, W, C). The feature vectors associated with each point
            in geom_feats.

        Returns
        -------
        final : torch.Tensor
            A tensor of shape (B, C*Z, X, Y), where:
            - Z is the discretized size in self.nx[2],
            - X, Y are from self.nx[0], self.nx[1],
            - The channel dimension is C*Z due to collapsing the Z dimension.

        Steps
        -----
        1) Flatten x to (Nprime, C) and geom_feats to (Nprime, 3) (plus we append batch indices).
        2) Discretize geom_feats into voxel indices by subtracting an offset, dividing by dx, and casting to int.
        3) Filter out-of-bounds indices.
        4) Sort by a 'rank' so that points belonging to the same voxel are consecutive.
        5) Use a cumsum trick or specialized function to pool (sum/max) features within each voxel.
        6) Create a voxel grid (B, C, Z, X, Y) and place the pooled features at the correct positions.
        7) Finally, collapse the Z dimension by concatenating along the channel axis, yielding (B, C*Z, X, Y).
        """
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # 1) Flatten feature tensor
        x = x.reshape(Nprime, C)

        # 2) Discretize geometry into voxel indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)

        # 3) Add batch index
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # (Nprime, 4)

        # 4) Filter out-of-bounds
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # 5) Sort by voxel rank
        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)
            + geom_feats[:, 1] * (self.nx[2] * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # 6) Pool features in the same voxel
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # 7) Create voxel grid (B, C, Z, X, Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[
            geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]
        ] = x

        # 8) Collapse Z dimension -> (B, C*Z, X, Y)
        final = torch.cat(final.unbind(dim=2), dim=1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
