# Codes are taken from BPNet, CVPR'21
# https://github.com/wbhu/BPNet/blob/main/dataset/voxelizer.py

import collections
import numpy as np
from lib.openscene.voxelization_utils import sparse_quantize
from scipy.linalg import expm, norm


# Rotation matrix along axis with angle theta
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class Voxelizer:

    def __init__(self,
                 voxel_size=1,
                 ignore_label=255):
        '''
        Args:
          voxel_size: side length of a voxel
          ignore_label: label assigned for ignore (not a training label).
        '''
        self.voxel_size = voxel_size
        self.ignore_label = ignore_label

    def get_transformation_matrix(self):
        voxelization_matrix = np.eye(4)
        # Transform pointcloud coordinate to voxel coordinate.
        scale = 1 / self.voxel_size
        np.fill_diagonal(voxelization_matrix[:3, :3], scale)
        return voxelization_matrix

    def voxelize(self, coords, feats, labels, center=None, link=None, return_ind=False):
        assert coords.shape[1] == 3 and coords.shape[0] == feats.shape[0] and coords.shape[0]

        # Get voxelization matrix
        rigid_transformation = self.get_transformation_matrix()

        homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
        coords_aug = np.floor(homo_coords @ rigid_transformation.T[:, :3])

        # assert coords_aug.min(0) is all zero in all three channels
        assert coords_aug.min(0).sum() == 0, 'Minimum of coordinates are not zeros!'
        inds, inds_reconstruct = sparse_quantize(coords_aug, return_index=True)

        coords_aug, feats, labels = coords_aug[inds], feats[inds], labels[inds]

        if return_ind:
            return coords_aug, feats, labels, np.array(inds_reconstruct), inds
        if link is not None:
            return coords_aug, feats, labels, np.array(inds_reconstruct), link[inds]

        return coords_aug, feats, labels, np.array(inds_reconstruct)
