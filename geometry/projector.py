# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""A collection of projection utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from lighthouse.geometry import sampling


def inv_depths(start_depth, end_depth, num_depths):
  """Returns reversed, sorted inverse interpolated depths.

  Args:
    start_depth: The first depth.
    end_depth: The last depth.
    num_depths: The total number of depths to create, include start_depth and
      end_depth are always included and other depths are interpolated between
      them, in inverse depth space.

  Returns:
    The depths sorted in descending order (so furthest first). This order is
    useful for back to front compositing.
  """

  depths = 1.0 / tf.linspace(1.0 / end_depth, 1.0 / start_depth, num_depths)
  return depths


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.

  Args:
    depth: [batch, height, width]
    pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
    intrinsics: camera intrinsics [batch, 3, 3]
    is_homogeneous: return in homogeneous coordinates

  Returns:
    Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
  """

  # Derived from code written by Tinghui Zhou and Shubham Tulsiani
  batch = tf.shape(depth)[0]
  height = tf.shape(depth)[1]
  width = tf.shape(depth)[2]
  depth = tf.reshape(depth, [batch, 1, -1])
  pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
  cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
  if is_homogeneous:
    ones = tf.ones([batch, 1, height * width])
    cam_coords = tf.concat([cam_coords, ones], axis=1)
  cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
  return cam_coords


def cam2pixel(cam_coords, proj):
  """Transforms coordinates in a camera frame to the pixel frame.

  Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]

  Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
  """

  # Derived from code written by Tinghui Zhou and Shubham Tulsiani
  batch = tf.shape(cam_coords)[0]
  height = tf.shape(cam_coords)[2]
  width = tf.shape(cam_coords)[3]
  cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
  unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
  x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
  y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
  z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
  x_n = x_u / (z_u + 1e-10)
  y_n = y_u / (z_u + 1e-10)
  pixel_coords = tf.concat([x_n, y_n], axis=1)
  pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
  return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])


def mpi_resample_cube(mpi, tgt, intrinsics, depth_planes, side_length,
                      cube_res):
  """Resample MPI onto cube centered at target point.

  Args:
    mpi: [B,H,W,D,C], input MPI
    tgt: [B,3], [x,y,z] coordinates for cube center (in reference/mpi frame)
    intrinsics: [B,3,3], MPI reference camera intrinsics
    depth_planes: [D] depth values for MPI planes
    side_length: metric side length of cube
    cube_res: resolution of each cube dimension

  Returns:
    resampled: [B, cube_res, cube_res, cube_res, C]
  """

  batch_size = tf.shape(mpi)[0]
  num_depths = tf.shape(mpi)[3]

  # compute MPI world coordinates
  intrinsics_tile = tf.tile(intrinsics, [num_depths, 1, 1])

  # create cube coordinates
  b_vals = tf.to_float(tf.range(batch_size))
  x_vals = tf.linspace(-side_length / 2.0, side_length / 2.0, cube_res)
  y_vals = tf.linspace(-side_length / 2.0, side_length / 2.0, cube_res)
  z_vals = tf.linspace(side_length / 2.0, -side_length / 2.0, cube_res)
  b, y, x, z = tf.meshgrid(b_vals, y_vals, x_vals, z_vals, indexing='ij')

  x = x + tgt[:, 0, tf.newaxis, tf.newaxis, tf.newaxis]
  y = y + tgt[:, 1, tf.newaxis, tf.newaxis, tf.newaxis]
  z = z + tgt[:, 2, tf.newaxis, tf.newaxis, tf.newaxis]

  ones = tf.ones_like(x)
  coords = tf.stack([x, y, z, ones], axis=1)
  coords_r = tf.reshape(
      tf.transpose(coords, [0, 4, 1, 2, 3]),
      [batch_size * cube_res, 4, cube_res, cube_res])

  # store elements with negative z vals for projection
  bad_inds = tf.less(z, 0.0)

  # project into reference camera to transform coordinates into MPI indices
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch_size * cube_res, 1, 1])
  intrinsics_tile = tf.tile(intrinsics, [cube_res, 1, 1])
  intrinsics_tile_4 = tf.concat(
      [intrinsics_tile,
       tf.zeros([batch_size * cube_res, 3, 1])], axis=2)
  intrinsics_tile_4 = tf.concat([intrinsics_tile_4, filler], axis=1)
  coords_proj = cam2pixel(coords_r, intrinsics_tile_4)
  coords_depths = tf.transpose(coords_r[:, 2:3, :, :], [0, 2, 3, 1])
  coords_depth_inds = (tf.to_float(num_depths) - 1) * (
      (1.0 / coords_depths) -
      (1.0 / depth_planes[0])) / ((1.0 / depth_planes[-1]) -
                                  (1.0 / depth_planes[0]))
  coords_proj = tf.concat([coords_proj, coords_depth_inds], axis=3)
  coords_proj = tf.transpose(
      tf.reshape(coords_proj, [batch_size, cube_res, cube_res, cube_res, 3]),
      [0, 2, 3, 1, 4])
  coords_proj = tf.concat([b[:, :, :, :, tf.newaxis], coords_proj], axis=4)

  # trilinear interpolation gather from MPI
  # interpolate pre-multiplied RGBAs, then un-pre-multiply
  mpi_alpha = mpi[Ellipsis, -1:]
  mpi_channels_p = mpi[Ellipsis, :-1] * mpi_alpha
  mpi_p = tf.concat([mpi_channels_p, mpi_alpha], axis=-1)

  resampled_p = sampling.trilerp_gather(mpi_p, coords_proj, bad_inds)

  resampled_alpha = tf.clip_by_value(resampled_p[Ellipsis, -1:], 0.0, 1.0)
  resampled_channels = resampled_p[Ellipsis, :-1] / (resampled_alpha + 1e-8)
  resampled = tf.concat([resampled_channels, resampled_alpha], axis=-1)

  return resampled, coords_proj


def spherical_cubevol_resample(vol, env2ref, cube_center, side_length, n_phi,
                               n_theta, n_r):
  """Resample cube volume onto spherical coordinates centered at target point.

  Args:
    vol: [B,H,W,D,C], input volume
    env2ref: [B,4,4], relative pose transformation (transform env to ref)
    cube_center: [B,3], [x,y,z] coordinates for center of cube volume
    side_length: side length of cube
    n_phi: number of samples along vertical spherical coordinate dim
    n_theta: number of samples along horizontal spherical coordinate dim
    n_r: number of samples along radius spherical coordinate dim

  Returns:
    resampled: [B, n_phi, n_theta, n_r, C]
  """

  batch_size = tf.shape(vol)[0]
  height = tf.shape(vol)[1]

  cube_res = tf.to_float(height)

  # create spherical coordinates
  b_vals = tf.to_float(tf.range(batch_size))
  phi_vals = tf.linspace(0.0, np.pi, n_phi)
  theta_vals = tf.linspace(1.5 * np.pi, -0.5 * np.pi, n_theta)

  # compute radii to use
  x_vals = tf.linspace(-side_length / 2.0, side_length / 2.0,
                       tf.to_int32(cube_res))
  y_vals = tf.linspace(-side_length / 2.0, side_length / 2.0,
                       tf.to_int32(cube_res))
  z_vals = tf.linspace(side_length / 2.0, -side_length / 2.0,
                       tf.to_int32(cube_res))
  y_c, x_c, z_c = tf.meshgrid(y_vals, x_vals, z_vals, indexing='ij')
  x_c = x_c + cube_center[:, 0, tf.newaxis, tf.newaxis, tf.newaxis]
  y_c = y_c + cube_center[:, 1, tf.newaxis, tf.newaxis, tf.newaxis]
  z_c = z_c + cube_center[:, 2, tf.newaxis, tf.newaxis, tf.newaxis]
  cube_coords = tf.stack([x_c, y_c, z_c], axis=4)
  min_r = tf.reduce_min(
      tf.norm(
          cube_coords -
          env2ref[:, :3, 3][:, tf.newaxis, tf.newaxis, tf.newaxis, :],
          axis=4),
      axis=[0, 1, 2, 3])  # side_length / cube_res
  max_r = tf.reduce_max(
      tf.norm(
          cube_coords -
          env2ref[:, :3, 3][:, tf.newaxis, tf.newaxis, tf.newaxis, :],
          axis=4),
      axis=[0, 1, 2, 3])

  r_vals = tf.linspace(max_r, min_r, n_r)
  b, phi, theta, r = tf.meshgrid(
      b_vals, phi_vals, theta_vals, r_vals,
      indexing='ij')  # currently in env frame

  # transform spherical coordinates into cartesian
  # (currently in env frame, z points forwards)
  x = r * tf.cos(theta) * tf.sin(phi)
  z = r * tf.sin(theta) * tf.sin(phi)
  y = r * tf.cos(phi)

  # transform coordinates into ref frame
  sphere_coords = tf.stack([x, y, z, tf.ones_like(x)], axis=-1)[Ellipsis, tf.newaxis]
  sphere_coords_ref = tfmm(env2ref, sphere_coords)
  x = sphere_coords_ref[Ellipsis, 0, 0]
  y = sphere_coords_ref[Ellipsis, 1, 0]
  z = sphere_coords_ref[Ellipsis, 2, 0]

  # transform coordinates into vol indices
  x_inds = (x - cube_center[:, 0, tf.newaxis, tf.newaxis, tf.newaxis] +
            side_length / 2.0) * ((cube_res - 1) / side_length)
  y_inds = -(y - cube_center[:, 1, tf.newaxis, tf.newaxis, tf.newaxis] -
             side_length / 2.0) * ((cube_res - 1) / side_length)
  z_inds = -(z - cube_center[:, 2, tf.newaxis, tf.newaxis, tf.newaxis] -
             side_length / 2.0) * ((cube_res - 1) / side_length)
  sphere_coords_inds = tf.stack([b, x_inds, y_inds, z_inds], axis=-1)

  # trilinear interpolation gather from volume
  # interpolate pre-multiplied RGBAs, then un-pre-multiply
  vol_alpha = tf.clip_by_value(vol[Ellipsis, -1:], 0.0, 1.0)
  vol_channels_p = vol[Ellipsis, :-1] * vol_alpha
  vol_p = tf.concat([vol_channels_p, vol_alpha], axis=-1)

  resampled_p = sampling.trilerp_gather(vol_p, sphere_coords_inds)

  resampled_alpha = resampled_p[Ellipsis, -1:]
  resampled_channels = resampled_p[Ellipsis, :-1] / (resampled_alpha + 1e-8)
  resampled = tf.concat([resampled_channels, resampled_alpha], axis=-1)

  return resampled, r_vals


def over_composite(rgbas):
  """Combines a list of rgba images using the over operation.

  Combines RGBA images from back to front (where back is index 0 in list)
  with the over operation.

  Args:
    rgbas: A list of rgba images, these are combined from *back to front*.

  Returns:
    Returns an RGB image.
  """

  alphas = rgbas[:, :, :, :, -1:]
  colors = rgbas[:, :, :, :, :-1]
  transmittance = tf.cumprod(
      1.0 - alphas + 1.0e-8, axis=3, exclusive=True, reverse=True) * alphas
  output = tf.reduce_sum(transmittance * colors, axis=3)
  accum_alpha = tf.reduce_sum(transmittance, axis=3)

  return tf.concat([output, accum_alpha], axis=3)


def interleave_shells(shells, radii):
  """Interleave spherical shell tensors out-to-in by radii."""

  radius_order = tf.argsort(radii, direction='DESCENDING')
  shells_interleaved = tf.gather(shells, radius_order, axis=3)
  return shells_interleaved


##########################
#  Homography/matrix math for MPIs and plane sweep volumes
#  From Ben Mildenhall at
#  https://github.com/Fyusion/LLFF/blob/master/llff/math/mpi_math.py
##########################

ALPHA_EPS = 1e-8


# tf.matmul seems to cause NaNs in some cases
# (maybe a perfect storm of GPU+env?)
def tfmm(A, B):
  """Redefined tensorflow matrix multiply."""

  with tf.variable_scope('tfmm'):
    return tf.reduce_sum(
        A[..., :, :, tf.newaxis] * B[..., tf.newaxis, :, :], axis=-2)


def myposes2mats(poses, fix_yx=False):
  """Converts my pose format into 4x4 extrinsic and 3x3 intrinsic matrices."""
  #  My rotations are in [down, right, backwards] orientation,
  #  hence the 'fix_yx' thing to convert from that format [-y x z]
  #  to the more conventional [x y z]

  with tf.variable_scope('myposes2mats'):

    def cat(arrs, ax):
      return tf.concat(arrs, ax)

    def stk(arrs, ax):
      return tf.stack(arrs, ax)

    c2w = poses[..., :3, :4] + 0.
    bottom0 = tf.zeros_like(c2w[..., :1, :3])
    bottom1 = tf.ones_like(c2w[..., :1, 3:4])
    bottom = tf.concat([bottom0, bottom1], -1)
    # fix the -y x thing
    if fix_yx:
      c2w = cat([c2w[..., :3, 1:2], -c2w[..., :3, 0:1], c2w[..., :3, 2:]], -1)

    R = c2w[..., :3, :3] + 0.
    t = c2w[..., :3, 3:4] + 0.
    T_c2w = cat([c2w, bottom], -2)

    N = len(R.get_shape().as_list())
    perm = list(range(N - 2)) + [N - 1, N - 2]
    R_inv = tf.transpose(R, perm)
    t_inv = -tfmm(R_inv, t)
    T_w2c = cat([cat([R_inv, t_inv], -1), bottom], -2)

    h, w, f = poses[..., 0, -1], poses[..., 1, -1], poses[..., 2, -1]
    m_z, m_o = tf.zeros_like(h), tf.ones_like(h)

    sh = tf.shape(poses)[:-2]
    sh = tf.concat([sh, tf.constant([3]), tf.constant([3])], -1)
    with tf.variable_scope('Kstuff'):
      K = stk([f, m_z, -w * .5, m_z, f, -h * .5, m_z, m_z, -m_o], -1)
      K = tf.reshape(K, sh)

      K_inv = stk(
          [1. / f, m_z, -w * .5 / f, m_z, 1. / f, -h * .5 / f, m_z, m_z, -m_o],
          -1)
      K_inv = tf.reshape(K_inv, sh)

    T_c2w, T_w2c, K, K_inv = map(lambda x: tf.cast(x, dtype=poses.dtype),
                                 [T_c2w, T_w2c, K, K_inv])
    return T_c2w, T_w2c, K, K_inv


def plane_homogs(pose_t,
                 pose_s,
                 depths,
                 planes_from_t=True,
                 y_flip=True,
                 fix_yx=False):
  """To warp a single plane, for PSV creation and MPI rendering."""

  with tf.variable_scope('plane_homogs'):
    T_t2w, _, _, K_t_inv = myposes2mats(pose_t, fix_yx=fix_yx)
    _, T_w2s, K_s, _ = myposes2mats(pose_s, fix_yx=fix_yx)

    T_t2s = tfmm(T_w2s, T_t2w)
    R = T_t2s[..., tf.newaxis, :3, :3]
    t = T_t2s[..., tf.newaxis, :3, 3:4]
    n = tf.constant(np.array([0, 0, 1.]), dtype=T_t2s.dtype)
    n = tf.reshape(n, [3, 1])
    nT = tf.transpose(n)
    a = tf.reshape(depths, [-1, 1, 1])

    if planes_from_t:
      H = R - tfmm(t, nT) / a
    else:
      H = R - tfmm(t, tfmm(nT, R)) / (a + tfmm(nT, t))

    H = tfmm(K_s[..., tf.newaxis, :, :], tfmm(H, K_t_inv[...,
                                                         tf.newaxis, :, :]))
    if y_flip:
      premat = tf.stack([1, 0, 0., 0, -1, pose_t[0, -1] - 1, 0, 0, 1.], -1)
      premat = tf.cast(premat, dtype=T_t2s.dtype)
      premat = tf.reshape(premat, [3, 3])

      postmat = tf.stack([1, 0, 0., 0, -1, pose_s[0, -1] - 1, 0, 0, 1.], -1)
      postmat = tf.cast(postmat, dtype=T_t2s.dtype)
      postmat = tf.reshape(postmat, [3, 3])

      H = tfmm(postmat, tfmm(H, premat))
    return H


def homog_warp(img, H, retcos=False, window=None):
  """Clone of tf.contrib.image.transform (fails on the RTX 2080 Ti)."""

  with tf.variable_scope('homog_warp'):
    sh = tf.shape(img)
    h, w = sh[-3], sh[-2]

    H = tf.reshape(tf.concat([H, tf.ones_like(H[:, :1])], -1), [-1, 3, 3])

    bds = tf.stack([0, 0, h, w], 0)
    if window is not None:
      bds = tf.cond(window[3] > 0, lambda: window, lambda: bds)
    coords = tf.meshgrid(
        tf.range(bds[0], bds[2]), tf.range(bds[1], bds[3]), indexing='ij')
    coords = tf.cast(tf.stack([coords[1], coords[0]], 0), H.dtype)  # [2, H, W]

    coords_t = tf.concat([coords, tf.ones_like(coords[:1, ...])],
                         0)  # [3, H, W]
    coords_t = tf.reshape(H, [-1, 3, 3, 1, 1]) * coords_t  # [-1, 3, 3, H, W]
    coords_t = tf.reduce_sum(coords_t, -3)  # [-1, 3, H, W]
    coords_t = coords_t[..., :2, :, :] / coords_t[...,
                                                  -1:, :, :]  # [-1, 2, H, W]

    warp = tf.transpose(coords_t, [0, 2, 3, 1])  # [-1, H, W, 2]
    # rect_tf = tf.squeeze(tf.contrib.resampler.resampler(img, warp))
    rect_tf = tf.squeeze(sampling.bilerp_gather(img, warp))

    rets = rect_tf
    if retcos:
      rets = [rect_tf, coords_t]
    return rets


def render_mpi_homogs(mpi_rgba,
                      pose,
                      newpose,
                      min_disp,
                      max_disp,
                      num_depths,
                      debug=False):
  """Render view at newpose from MPI at pose."""

  with tf.variable_scope('render_mpi_homogs'):

    outs = {}
    dispvals = tf.linspace(min_disp, max_disp, num_depths)

    H = plane_homogs(
        newpose,
        pose,
        1. / dispvals,
        planes_from_t=False,
        y_flip=True,
        fix_yx=False)
    H = tf.reshape(H, [-1, 9])
    H = H[:, :8] / H[:, 8:]

    data_in = tf.transpose(tf.squeeze(mpi_rgba), [2, 0, 1, 3])
    window = tf.cast([0, 0, newpose[0, -1], newpose[1, -1]], tf.int32)
    mpi_reproj = homog_warp(data_in, H, window=window)
    mpi_reproj = tf.transpose(mpi_reproj, [1, 2, 0, 3])[tf.newaxis, ...]

    # back to front compositing
    mpiR_alpha = mpi_reproj[..., 3:4]  # 1 H W D 1
    mpiR_color = mpi_reproj[..., 0:3]  # 1 H W D 3

    # Add small ALPHA_EPS to prevent gradient explosion nans from tf.cumprod
    weights = mpiR_alpha * tf.cumprod(
        1. - mpiR_alpha + ALPHA_EPS, -2, exclusive=True, reverse=True)
    alpha_acc = tf.reduce_sum(weights[..., 0], -1)
    rendering = tf.reduce_sum(weights * mpiR_color, -2)
    accum = tf.cumsum(weights * mpiR_color, -2, reverse=True)

    return rendering, alpha_acc, accum


def make_psv_homogs(img, pose, newpose, dispvals, num_depths, window=None):
  """Create a plane sweep volume, vectorized on initial axes of img and pose."""

  with tf.variable_scope('make_cv_homogs'):

    H = plane_homogs(
        newpose,
        pose,
        1. / dispvals,
        planes_from_t=True,
        y_flip=True,
        fix_yx=False)  # [N, D, 3, 3]
    H = tf.reshape(H, [-1, 9])  # [N*D, 9]
    H = H[:, :8] / H[:, 8:]

    img_sh = tf.shape(tf.squeeze(img))
    img = tf.reshape(
        img, [-1, 1, img_sh[-3], img_sh[-2], img_sh[-1]])  # [N, 1, H, W, 3]
    img_tiled = tf.tile(img,
                        [1, num_depths, 1, 1, 1])  # go in as [N, D, H, W, 3]
    img_tiled = tf.reshape(
        img_tiled, [-1, img_sh[-3], img_sh[-2], img_sh[-1]])  # [N*D, H, W, 3]

    cvd = homog_warp(img_tiled, H, window=window)  # come out as [N*D, H, W, 3]
    h, w = img_sh[-3], img_sh[-2]
    if window is not None:
      h = tf.cond(window[3] > 0, lambda: window[2] - window[0], lambda: h)
      w = tf.cond(window[3] > 0, lambda: window[3] - window[1], lambda: w)
    cvd = tf.reshape(cvd, [-1, num_depths, h, w, img_sh[-1]])  # [N, D, H, W, 3]
    cvd = tf.squeeze(tf.transpose(
        cvd, [2, 3, 1, 4, 0]))  # [H, W, D, 3, N] or [H, W, D, 3]

    return cvd
