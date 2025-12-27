import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)



class WireDatasetConfig(object):
    def __init__(self):
        # ------------------------
        # BASIC DATASET SETTINGS
        # ------------------------
        self.num_class = 1                # only wire
        self.num_heading_bin = 1          # no rotation needed
        self.num_size_cluster = 1         # one size template

        # ------------------------
        # CLASS MAPPING
        # ------------------------
        self.type2class = {'wire': 0}
        self.class2type = {0: 'wire'}
        self.type2onehotclass = {'wire': 0}

        # ------------------------
        # MEAN SIZE (l, w, h)
        # ------------------------
        # IMPORTANT:
        # Replace this with the mean of your training boxes later
        self.type_mean_size = {
            'wire': np.array([0.10, 0.10, 1.00], dtype=np.float32)
        }

        self.mean_size_arr = np.zeros((self.num_size_cluster, 3), dtype=np.float32)
        self.mean_size_arr[0, :] = self.type_mean_size['wire']

    # -------------------------------------------------------
    # SIZE FUNCTIONS
    # -------------------------------------------------------
    def size2class(self, size, type_name):
        """
        Convert box size (l,w,h) to size class and residual
        """
        size_class = 0
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        """
        Inverse of size2class
        """
        mean_size = self.type_mean_size[self.class2type[int(pred_cls)]]
        return mean_size + residual

    # -------------------------------------------------------
    # ANGLE FUNCTIONS (KEPT FOR COMPATIBILITY)
    # -------------------------------------------------------
    def angle2class(self, angle):
        """
        Dummy angle handling (no rotation)
        """
        return 0, 0.0

    def class2angle(self, pred_cls, residual, to_label_format=True):
        return 0.0

    # -------------------------------------------------------
    # PARAM → OBB
    # -------------------------------------------------------
    def param2obb(self, center, heading_class, heading_residual,
                  size_class, size_residual):
        """
        Convert network output to oriented bounding box
        (axis-aligned in your case)
        """
        box_size = self.class2size(int(size_class), size_residual)

        obb = np.zeros((7,), dtype=np.float32)
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = 0.0  # no rotation

        return obb
