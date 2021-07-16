# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp

from ...utils import shape_util
from .base_method import AugmentationMethod


class FreqMasking(AugmentationMethod):
    def __init__(self, num_masks: int = 1, mask_factor: float = 27):
        self.num_masks = num_masks
        self.mask_factor = mask_factor

    @tf.function
    def augment(self, spectrogram: tf.Tensor):
        """
        Masking the frequency channels (shape[1])
        Args:
            spectrogram: shape (T, num_feature_bins, V)
        Returns:
            frequency masked spectrogram
        """
        T, F, V = shape_util.shape_list(spectrogram, out_type=tf.int32)
        for _ in range(self.num_masks):
            f = tf.random.uniform([], minval=0, maxval=self.mask_factor, dtype=tf.int32)
            f = tf.minimum(f, F)
            f0 = tf.random.uniform([], minval=0, maxval=(F - f), dtype=tf.int32)
            mask = tf.concat([
                tf.ones([T, f0, V], dtype=spectrogram.dtype),
                tf.zeros([T, f, V], dtype=spectrogram.dtype),
                tf.ones([T, F - f0 - f, V], dtype=spectrogram.dtype)
            ], axis=1)
            spectrogram = spectrogram * mask
        return spectrogram


class TimeMasking(AugmentationMethod):
    def __init__(self, num_masks: int = 1, mask_factor: float = 100, p_upperbound: float = 1.0):
        self.num_masks = num_masks
        self.mask_factor = mask_factor
        self.p_upperbound = p_upperbound

    @tf.function
    def augment(self, spectrogram: tf.Tensor):
        """
        Masking the time channel (shape[0])
        Args:
            spectrogram: shape (T, num_feature_bins, V)
        Returns:
            frequency masked spectrogram
        """
        T, F, V = shape_util.shape_list(spectrogram, out_type=tf.int32)
        for _ in range(self.num_masks):
            t = tf.random.uniform([], minval=0, maxval=self.mask_factor, dtype=tf.int32)
            t = tf.minimum(t, tf.cast(tf.cast(T, dtype=tf.float32) * self.p_upperbound, dtype=tf.int32))
            t0 = tf.random.uniform([], minval=0, maxval=(T - t), dtype=tf.int32)
            mask = tf.concat([
                tf.ones([t0, F, V], dtype=spectrogram.dtype),
                tf.zeros([t, F, V], dtype=spectrogram.dtype),
                tf.ones([T - t0 - t, F, V], dtype=spectrogram.dtype)
            ], axis=0)
            spectrogram = spectrogram * mask
        return spectrogram


class TimeWarp(AugmentationMethod):
    def __init__(self, warp_factor: float = 80):
        self.warp_factor = warp_factor

    @tf.function
    def augment(self, spectrogram: tf.Tensor):
        """
        Warping the time channel (shape[0])
        Args:
            spectrogram: shape (T, num_feature_bins, V)
        Returns:
            frequency masked spectrogram
        """
        tau, height, _ = shape_util.shape_list(spectrogram, out_type=tf.int32)
        if tau <= 2 * self.warp_factor:
            tf.print("Spectrogram too short (len = ", tau, ") for time_warping_para`m = ", self.warp_factor,
                     ". Skipping warping.")
            return spectrogram
        generator = tf.random.get_global_generator()
        center_height = height / 2

        with tf.name_scope('warping'):
            random_point = generator.uniform(minval=self.warp_factor, maxval=tau - self.warp_factor, dtype=tf.int32,
                                             shape=(),
                                             name='get_random_point')
            w = generator.uniform(minval=0, maxval=self.warp_factor, dtype=tf.int32, shape=(), name='get_warping_factor')

            control_point_locations = tf.convert_to_tensor([[[random_point, center_height],
                                                             [0, center_height],
                                                             [tau, center_height]]],
                                                           dtype=tf.float32)

            control_point_destination = tf.convert_to_tensor([[[random_point + w, center_height],
                                                               [0, center_height],
                                                               [tau, center_height]]],
                                                             dtype=tf.float32)
            spectrogram, _ = sparse_image_warp(spectrogram,
                                       source_control_point_locations=control_point_locations,
                                       dest_control_point_locations=control_point_destination,
                                       interpolation_order=2,
                                       regularization_weight=0,
                                       num_boundary_points=1
                                       )

        return spectrogram