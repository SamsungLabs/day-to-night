"""
Author(s):
Abdelrahman Abdelhamed (a.abdelhamed@samsung.com)

Copyright (c) 2022 Samsung Electronics Co., Ltd.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import numpy as np

def RGGB2Bayer(im, _cfa_pattern=[0,1,1,2]):
    # add 1 to the second-green and the blue channel (e.g., [0, 1, 1, 2] will be [0, 1, 2, 3])
    _cfa_pattern_arr=np.asarray(_cfa_pattern)
    _cfa_pattern_arr[_cfa_pattern_arr == 2] += 1
    _cfa_pattern_arr[2:][_cfa_pattern_arr[2:] == 1] += 1
    # convert RGGB stacked image to one channel Bayer
    bayer = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    bayer[0::2, 0::2] = im[:, :, _cfa_pattern_arr[0]]
    bayer[0::2, 1::2] = im[:, :, _cfa_pattern_arr[1]]
    bayer[1::2, 0::2] = im[:, :, _cfa_pattern_arr[2]]
    bayer[1::2, 1::2] = im[:, :, _cfa_pattern_arr[3]]
    return bayer

def stack_rggb_channels(raw_image, bayer_pattern=None):
    """
    Stack the four channels of a CFA/Bayer raw image along a third dimension.
    """
    if bayer_pattern is None:
        bayer_pattern = [0, 1, 1, 2]
    height, width = raw_image.shape
    channels = []
    pattern = np.array(bayer_pattern)
    # add 1 to the second-green and the blue channel (e.g., [0, 1, 1, 2] will be [0, 1, 2, 3])
    pattern[pattern == 2] += 1
    pattern[2:][pattern[2:] == 1] += 1
    idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for c in pattern:
        raw_image_c = raw_image[idx[c][0]:height:2, idx[c][1]:width:2].copy()
        channels.append(raw_image_c)
    channels = np.stack(channels, axis=-1)
    return channels


def stack_rgb_channels(raw_image, bayer_pattern):
    """
    Stack the four channels in a CFA/Bayer image into 3 RGB channels, averaging the two G channels.
    """
    rggb = stack_rggb_channels(raw_image, bayer_pattern)
    rgb = np.zeros((rggb.shape[0], rggb.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = rggb[:, :, 0]
    rgb[:, :, 1] = (rggb[:, :, 1] + rggb[:, :, 2]) / 2.0
    rgb[:, :, 2] = rggb[:, :, 3]
    return rgb


def rggb_to_rgb(image_4ch, bayer_pattern):
    bayer_pattern = list(bayer_pattern)
    g1_idx = bayer_pattern.index(1)
    g2_idx = 3 if g1_idx == 0 else 2
    r_idx = bayer_pattern.index(0)
    b_idx = bayer_pattern.index(2)
    g = np.mean([image_4ch[:, :, g1_idx], image_4ch[:, :, g2_idx]], axis=0)
    rgb = np.stack([image_4ch[:, :, r_idx], g, image_4ch[:, :, b_idx]], axis=-1)
    return rgb
