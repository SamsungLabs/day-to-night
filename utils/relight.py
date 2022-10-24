"""
Author(s):

Abdelrahman Abdelhamed (a.abdelhamed@samsung.com)
Abhijith Punnappurath (abhijith.p@samsung.com)

Copyright (c) 2022 Samsung Electronics Co., Ltd.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.


Synthetically relighting day-to-night images.
"""

import cv2
import numpy as np
from pipeline.pipeline_utils import white_balance


class LocalLight:
    """
    Local light source.
    """

    def __init__(self, id_, color, location, size, scale, ambient=False, sat=False):
        self.id = id_
        self.color = color
        self.location = location  # (y, x)
        self.size = size  # (h, w)
        self.scale = scale  # a scale factor to be applied to the illuminant
        self.ambient = ambient  # whether the light is applied uniformly over the whole image
        self.sat = sat  # whether the light is saturated

    def get_gaussian_kernel(self):
        return gaussian_kernel(self.size[0], self.size[1], self.sat)

    def get_translated_mask(self, shape):
        if self.ambient:
            translated_mask = np.ones(shape, dtype=np.float32)
        else:
            translated_mask = translate_in_frame(self.get_gaussian_kernel(), self.location[0], self.location[1], shape)
        return translated_mask


def relight_locally(image, illuminants, cfa_pattern, clip=True, invert_wb=True, min_light_size=0.5, max_light_size=1.0, num_sat_lights=5):
    """
    Relight image with multiple locally-variant illuminants.
    :param image: Input image in [0, 1].
    :param illuminants: List or array of illuminant vectors.
    :param cfa_pattern: CFA/Bayer pattern.
    :param clip: Whether to clip values below zero. Values above 1 are always clipped.
    :param invert_wb: Whether to inverse illuminant vector.
    :param min_light_size: Minimum size of local light, as a percentage of image dimensions.
    :param max_light_size: Maximum size of local light, as a percentage of image dimensions.
    :param num_sat_lights: number of small saturated local lights.
    :return: Locally relit image.
    """

    # generate local lights
    local_lights = []
    for i in range(len(illuminants)):
        light = generate_random_light(id_=i, illuminant=illuminants[i], image_shape=image.shape,
                                      min_light_size=min_light_size, max_light_size=max_light_size, scale=1.0,
                                      ambient=i == 0, sat=i>=len(illuminants)-num_sat_lights)  # first light is ambient, last num_sat_lights are saturated
        local_lights.append(light)

    # first scaled ambient light is applied
    local_lights[0].scale = 0.05

    # first light is a special one, serves as an ambient light over the whole image with a small scaling factor
    # mask is all ones, scale is a small number (e.g., 0.05)

    # apply local lights
    image_relight = apply_local_lights(image, local_lights, cfa_pattern, clip, invert_wb, num_sat_lights)

    return image_relight, local_lights


def apply_local_lights(image, local_lights, cfa_pattern, clip=True, invert_wb=True, num_sat_lights=5):
    """
    Apply a list of local lights to image.
    :param image: Input raw image in [0, 1].
    :param local_lights: A list of LocalLight objects.
    :param cfa_pattern: CFA/Bayer pattern.
    :param clip: Whether to clip values below zero. Values above 1 are always clipped.
    :param invert_wb: Whether to inverse illuminant vector.
    :param num_sat_lights: number of small saturated local lights.

    :return: Locally relit image.
    """

    image_relights = []

    for light in local_lights:
        # relight with one local light
        illuminant = light.color
        if invert_wb:
            illuminant = list(1.0 / np.asarray(light.color) / light.scale)
        image_relight_1 = white_balance(image, illuminant, cfa_pattern, clip)

        if clip:
            image_relight_1[image_relight_1 < 0] = 0
        image_relight_1[image_relight_1 > 1] = 1

        image_relights.append(image_relight_1)

    # weighted average of original image and locally relit images
    weights = np.array([ll.get_translated_mask(image.shape) for ll in local_lights])  # for relit images

    image_relight = np.average(np.array(image_relights[:len(local_lights)-num_sat_lights]), axis=0, weights=weights[:len(local_lights)-num_sat_lights])
    for ll in range(len(local_lights)-num_sat_lights,len(local_lights)):
        image_relight += (50+50*np.random.rand())*weights[ll, :, :] * image_relights[ll]
    image_relight[image_relight > 1] = 1

    return image_relight


def apply_local_lights_rgb(image_rgb, local_lights, clip=True, invert_wb=True):
    """
    Apply a list of local lights to image.
    :param image_rgb: Input RGB 3-channel image in [0, 1].
    :param local_lights: A list of LocalLight objects.
    :param clip: Whether to clip values below zero. Values above 1 are always clipped.
    :param invert_wb: Whether to inverse illuminant vector.
    :return: Locally relit image.
    """

    image_relights = []

    for light in local_lights:
        # relight with one local light
        illuminant = light.color
        if invert_wb:
            illuminant = 1.0 / np.asarray(light.color)

        image_relight_1 = image_rgb / (illuminant / light.scale)[np.newaxis, np.newaxis, :]

        if clip:
            image_relight_1[image_relight_1 < 0] = 0
        image_relight_1[image_relight_1 > 1] = 1

        image_relights.append(image_relight_1)

    # weighted average of original image and locally relit images
    weights = np.array([
        np.tile(ll.get_translated_mask(image_rgb.shape[:2])[:, :, np.newaxis], [1, 1, 3])
        for ll in local_lights  # for relit images
    ])

    image_relight = np.average(np.array(image_relights), axis=0, weights=weights)

    return image_relight


def generate_random_light(id_, illuminant, image_shape, min_light_size=0.5, max_light_size=1.0, scale=1.0,
                          ambient=False, sat=False):
    """
    Generate a local light with random location and size.
    :param id_: ID.
    :param illuminant: Illuminant vector.
    :param image_shape: Target image shape.
    :param min_light_size: Minimum size of local light, as a percentage of image dimensions.
    :param max_light_size: Maximum size of local light, as a percentage of image dimensions.
    :param scale: A scale factor to be applied to the illuminant.
    :param ambient: Whether the light is applied uniformly over the whole image.
    :param sat: Whether the light is saturated.
    :return: LocalLight object.
    """
    light = LocalLight(
        id_=id_,
        color=illuminant,
        location=[
            np.random.randint(int(image_shape[0] * .1), int(image_shape[0] * .9)),  # y
            np.random.randint(int(image_shape[1] * .1), int(image_shape[1] * .9))],  # x
        size=[
            np.random.randint(int(image_shape[0] * min_light_size), int(image_shape[0] * max_light_size)),  # h
            np.random.randint(int(image_shape[1] * min_light_size), int(image_shape[1] * max_light_size))  # w
        ],
        scale=scale,
        ambient=ambient,
        sat=sat,
    )
    return light


def gaussian_kernel(h, w, sat):
    """
    Returns a Gaussian kernel with specified size (h, w).
    :param h: Height of Gaussian kernel.
    :param w: Width of Gaussian kernel.
    """
    sz = max(h, w)
    if sat:
        szv = sz / (5*np.random.rand()+5)
        gk = cv2.getGaussianKernel(sz, (0.3 * ((szv - 1) * 0.5 - 1) + 0.8) )
    else:
        gk = cv2.getGaussianKernel(sz, -1)
    gk = gk.T * gk
    gk /= gk.max()
    gk = cv2.resize(gk, dsize=(w, h))
    gk[gk < 1e-7] = 1e-7
    return gk


def translate_in_frame(arr, ty, tx, target_size_hw):
    """
    Translate a 2D array in a target frame size.
    :param arr: 2D array.
    :param ty: Y target location.
    :param tx: X target location.
    :param target_size_hw: Target frame size (h, w).
    :return: Translated array in the target frame.
    """
    translation = np.float32([
        [1, 0, tx - arr.shape[1] // 2],
        [0, 1, ty - arr.shape[0] // 2]
    ])
    translated_array = cv2.warpAffine(arr, translation, (target_size_hw[1], target_size_hw[0]))
    return translated_array.astype(np.float32)
