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


Camera pipeline utilities.
"""
import math
import os
import cv2
import numpy as np
import exifread
import rawpy
import struct
# from exifread import Ratio
from .exif_data_formats import exif_formats
from fractions import Fraction
from exifread.utils import Ratio
from scipy.io import loadmat
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
from .exif_utils import parse_exif_tag, parse_exif, get_tag_values_from_ifds
from .opcode import parse_opcode_lists
from .cct_utils import raw_rgb_to_cct, interpolate_cst


def get_visible_raw_image(image_path):
    raw_image = rawpy.imread(image_path).raw_image_visible.copy()
    return raw_image


def save_image_stage(image, dng_fn, stage, save_as):
    output_image_path = dng_fn.replace('.dng', '_{}.{}'.format(stage, save_as))
    output_image = (image * 255).astype(np.uint8)
    if len(output_image.shape) > 2:
        output_image = output_image[:, :, ::-1]
    if save_as == 'jpg':
        cv2.imwrite(output_image_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(output_image_path, output_image)


def get_image_tags(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
    return tags


def get_image_ifds(image_path):
    ifds = parse_exif(image_path, verbose=False)
    return ifds


def get_metadata(image_path):
    metadata = {}
    tags = get_image_tags(image_path)
    ifds = get_image_ifds(image_path)
    metadata['active_area'] = get_active_area(tags, ifds)
    metadata['linearization_table'] = get_linearization_table(tags, ifds)
    metadata['black_level'] = get_black_level(tags, ifds)
    metadata['white_level'] = get_white_level(tags, ifds)
    metadata['cfa_pattern'] = get_cfa_pattern(tags, ifds)
    metadata['as_shot_neutral'] = get_as_shot_neutral(tags, ifds)
    color_matrix_1, color_matrix_2 = get_color_matrices(tags, ifds)
    metadata['color_matrix_1'] = color_matrix_1
    metadata['color_matrix_2'] = color_matrix_2
    metadata['orientation'] = get_orientation(tags, ifds)
    metadata['noise_profile'] = get_noise_profile(tags, ifds)
    metadata['iso'] = get_iso(ifds)
    metadata['exposure_time'] = get_exposure_time(ifds)
    metadata['default_crop_origin'] = get_default_crop_origin(ifds)
    metadata['default_crop_size'] = get_default_crop_size(ifds)
    # ...

    # opcode lists
    metadata['opcode_lists'] = parse_opcode_lists(ifds)

    # fall back to default values, if necessary
    if metadata['black_level'] is None:
        metadata['black_level'] = 0
        print("Black level is None; using 0.")
    if metadata['white_level'] is None:
        metadata['white_level'] = 2 ** 16
        print("White level is None; using 2 ** 16.")
    if metadata['cfa_pattern'] is None:
        metadata['cfa_pattern'] = [0, 1, 1, 2]
        print("CFAPattern is None; using [0, 1, 1, 2] (RGGB)")
    if metadata['as_shot_neutral'] is None:
        metadata['as_shot_neutral'] = [1, 1, 1]
        print("AsShotNeutral is None; using [1, 1, 1]")
    if metadata['color_matrix_1'] is None:
        metadata['color_matrix_1'] = [1] * 9
        print("ColorMatrix1 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata['color_matrix_2'] is None:
        metadata['color_matrix_2'] = [1] * 9
        print("ColorMatrix2 is None; using [1, 1, 1, 1, 1, 1, 1, 1, 1]")
    if metadata['orientation'] is None:
        metadata['orientation'] = 0
        print("Orientation is None; using 0.")
    # ...
    return metadata


def get_active_area(tags, ifds):
    possible_keys = ['Image Tag 0xC68D', 'Image Tag 50829', 'ActiveArea', 'Image ActiveArea']
    return get_values(tags, possible_keys)


def get_linearization_table(tags, ifds):
    possible_keys = ['Image Tag 0xC618', 'Image Tag 50712', 'LinearizationTable', 'Image LinearizationTable']
    return get_values(tags, possible_keys)


def get_black_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61A', 'Image Tag 50714', 'BlackLevel', 'Image BlackLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("Black level not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(50714, ifds)
    return vals


def get_white_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61D', 'Image Tag 50717', 'WhiteLevel', 'Image WhiteLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("White level not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(50717, ifds)
    return vals


def get_cfa_pattern(tags, ifds):
    possible_keys = ['CFAPattern', 'Image CFAPattern']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("CFAPattern not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(33422, ifds)
    return vals


def get_as_shot_neutral(tags, ifds):
    possible_keys = ['Image Tag 0xC628', 'Image Tag 50728', 'AsShotNeutral', 'Image AsShotNeutral']
    return get_values(tags, possible_keys)


def get_color_matrices(tags, ifds):
    possible_keys_1 = ['Image Tag 0xC621', 'Image Tag 50721', 'ColorMatrix1', 'Image ColorMatrix1']
    color_matrix_1 = get_values(tags, possible_keys_1)
    possible_keys_2 = ['Image Tag 0xC622', 'Image Tag 50722', 'ColorMatrix2', 'Image ColorMatrix2']
    color_matrix_2 = get_values(tags, possible_keys_2)
    return color_matrix_1, color_matrix_2


def get_orientation(tags, ifds):
    possible_tags = ['Orientation', 'Image Orientation']
    return get_values(tags, possible_tags)


def get_noise_profile(tags, ifds):
    possible_keys = ['Image Tag 0xC761', 'Image Tag 51041', 'NoiseProfile', 'Image NoiseProfile']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("Noise profile not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(51041, ifds)
    return vals


def get_iso(ifds):
    # 0x8827	34855
    return get_tag_values_from_ifds(34855, ifds)


def get_exposure_time(ifds):
    # 0x829a	33434
    exposure_time = get_tag_values_from_ifds(33434, ifds)[0]
    return float(exposure_time.numerator) / float(exposure_time.denominator)


def get_default_crop_origin(ifds):
    return get_tag_values_from_ifds(50719, ifds)


def get_default_crop_size(ifds):
    return get_tag_values_from_ifds(50720, ifds)


def get_values(tags, possible_keys):
    values = None
    for key in possible_keys:
        if key in tags.keys():
            values = tags[key].values
    return values


def active_area_cropping(image, active_area):
    if active_area is not None and active_area != [0, 0, image.shape[0], image.shape[1]]:
        image = image[active_area[0]: active_area[2], active_area[1]:active_area[3]]

    return image


def default_cropping(image, default_crop_origin, default_crop_size):
    if default_crop_origin is not None and default_crop_size is not None:
        if type(default_crop_origin[0]) is Fraction:
            default_crop_origin = [float(x.numerator) / float(x.denominator) for x in default_crop_origin]
        if type(default_crop_size[0]) is Fraction:
            default_crop_size = [float(x.numerator) / float(x.denominator) for x in default_crop_size]
        if np.any([x != int(x) for x in default_crop_size]):
            raise ValueError('Default crop size is not integer, default_crop_size = {}'.format(default_crop_size))

        # when default_crop_origin and default_crop_size come in (H,W) rather than (W,H), flip the order
        if (default_crop_size[0] < default_crop_size[1] and image.shape[0] < image.shape[1]) or \
                (default_crop_size[0] > default_crop_size[1] and image.shape[0] > image.shape[1]):
            default_crop_size.reverse()
            default_crop_origin.reverse()

        # check if any elements in default crop origin or default crop size is not an integer
        if np.any([x != int(x) for x in default_crop_origin]):
            xs, ys = np.meshgrid(np.arange(default_crop_size[0]) + default_crop_origin[0],
                                 np.arange(default_crop_size[1]) + default_crop_origin[1])
            xs = xs.astype(np.float32)
            ys = ys.astype(np.float32)
            image = cv2.remap(image, xs, ys, cv2.INTER_LINEAR)
        else:
            image = image[int(default_crop_origin[1]):int(default_crop_origin[1] + default_crop_size[1]),
                    int(default_crop_origin[0]):int(default_crop_origin[0] + default_crop_size[0]), :]
    return image


def resize(image, target_size):
    """
    target_size: (width, height)
    """
    return cv2.resize(image, dsize=target_size, interpolation=cv2.INTER_LINEAR)


def normalize(raw_image, black_level, white_level, clip=True):
    if type(black_level) is list and len(black_level) == 1:
        black_level = float(black_level[0])
    if type(white_level) is list and len(white_level) == 1:
        white_level = float(white_level[0])
    black_level_mask = black_level
    if type(black_level) is list and len(black_level) == 4:
        if type(black_level[0]) is Ratio:
            black_level = ratios2floats(black_level)
        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]
    normalized_image = raw_image.astype(np.float32) - black_level_mask
    # if some values were smaller than black level
    if clip:
        normalized_image[normalized_image < 0] = 0
    normalized_image = normalized_image / (white_level - black_level_mask)
    return normalized_image


def denormalize(raw_image, black_level, white_level, clip=True):
    if type(black_level) is list and len(black_level) == 1:
        black_level = float(black_level[0])
    if type(white_level) is list and len(white_level) == 1:
        white_level = float(white_level[0])
    black_level_mask = black_level
    if type(black_level) is list and len(black_level) == 4:
        if type(black_level[0]) is Ratio:
            black_level = ratios2floats(black_level)
        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]
    denormalized_image = raw_image.astype(np.float32) * (white_level - black_level_mask) + black_level_mask
    # always clip when denormalizing, to [0, white_level]
    denormalized_image[denormalized_image < 0] = 0
    denormalized_image[denormalized_image > white_level] = white_level
    return denormalized_image


def ratios2floats(ratios):
    floats = []
    for ratio in ratios:
        floats.append(float(ratio.num) / ratio.den)
    return floats


def white_balance(normalized_image, as_shot_neutral, cfa_pattern, clip=True):
    if type(as_shot_neutral[0]) is Ratio:
        as_shot_neutral = ratios2floats(as_shot_neutral)
    idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    step2 = 2
    white_balanced_image = np.zeros(normalized_image.shape)
    for i, idx in enumerate(idx2by2):
        idx_y = idx[0]
        idx_x = idx[1]
        white_balanced_image[idx_y::step2, idx_x::step2] = \
            normalized_image[idx_y::step2, idx_x::step2] / as_shot_neutral[cfa_pattern[i]]
    if clip:
        white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)
    # alwyas clip at 1
    white_balanced_image[white_balanced_image > 1.0] = 1.0
    return white_balanced_image


def lens_distortion_correction(image, rect_warp_opcode, clip=True):
    # TODO This function does not work as expected on images tested, need to
    #  investigate whether it's caused by inaccurate metadata or the algorithm.
    # TODO currently we assume there is only one coeff set, need to extend this to 3
    result_image = np.zeros_like(image)

    num_planes = rect_warp_opcode.data['N']
    coeff_set = rect_warp_opcode.data['coefficient_set'][0]
    k_r0 = coeff_set['k_r0']
    k_r1 = coeff_set['k_r1']
    k_r2 = coeff_set['k_r2']
    k_r3 = coeff_set['k_r3']
    k_t0 = coeff_set['k_t0']
    k_t1 = coeff_set['k_t1']
    cx = rect_warp_opcode.data['cx']
    cy = rect_warp_opcode.data['cy']

    x0 = y0 = 0
    x1 = image.shape[1]
    y1 = image.shape[0]
    cx = x0 + cx * (x1 - x0)
    cy = y0 + cy * (y1 - y0)
    mx = max(abs(x0 - cx), abs(x1 - cx))
    my = max(abs(y0 - cy), abs(y1 - cy))
    m = math.sqrt(mx ** 2 + my ** 2)

    for y, row in enumerate(image):
        for x, col in enumerate(row):
            dx = (x - cx) / m
            dy = (y - cy) / m
            r = math.sqrt(dx ** 2 + dy ** 2)
            f = k_r0 + k_r1 * (r ** 2) + k_r2 * (r ** 4) + k_r3 * (r ** 6)
            dxr = f * dx
            dyr = f * dy
            dxt = k_t0 * 2 * dx * dy + k_t1 * (r ** 2 + 2 * (dx ** 2))
            dyt = k_t1 * 2 * dx * dy + k_t0 * (r ** 2 + 2 * (dy ** 2))
            x_ = cx + m * (dxr + dxt)
            y_ = cy + m * (dyr + dyt)
            for p in range(len(col)):
                # nearest neighbor

                # y_ = int(round(y_))
                # x_ = int(round(x_))
                #
                # y_ = max(y_, 0)
                # x_ = max(x_, 0)
                #
                # y_ = min(y_, image.shape[0] - 1)
                # x_ = min(x_, image.shape[1] - 1)
                #
                # result_image[y, x, p] = image[y_, x_, p]

                # bilinear interpolation

                x_ratio = x_ - math.floor(x_)
                p_y0 = x_ratio * image[math.floor(y_), math.ceil(x_), p] + (1 - x_ratio) * image[
                    math.floor(y_), math.floor(x_), p]
                p_y1 = x_ratio * image[math.ceil(y_), math.ceil(x_), p] + (1 - x_ratio) * image[
                    math.ceil(y_), math.floor(x_), p]
                y_ratio = y_ - math.floor(y_)
                pixel = y_ratio * p_y1 + (1 - y_ratio) * p_y0
                result_image[y, x, p] = pixel

    if clip:
        result_image = np.clip(result_image, 0.0, 1.0)
    return result_image


def lens_shading_correction(raw_image, gain_map_opcode, bayer_pattern, clip=True):
    """
    Apply lens shading correction map.
    :param raw_image: Input normalized (in [0, 1]) raw image.
    :param gain_map_opcode: Gain map opcode.
    :param bayer_pattern: Bayer pattern (RGGB, GRBG, ...).
    :param clip: Whether to clip result image to [0, 1].
    :return: Image with gain map applied; lens shading corrected.
    """

    gain_map = gain_map_opcode.data['map_gain_2d']

    # resize gain map, make it 4 channels, if needed
    gain_map = cv2.resize(gain_map, dsize=(raw_image.shape[1] // 2, raw_image.shape[0] // 2),
                          interpolation=cv2.INTER_LINEAR)
    if len(gain_map.shape) == 2:
        gain_map = np.tile(gain_map[..., np.newaxis], [1, 1, 4])

    # TODO: consider other parameters

    top = gain_map_opcode.data['top']
    left = gain_map_opcode.data['left']
    bottom = gain_map_opcode.data['bottom']
    right = gain_map_opcode.data['right']
    rp = gain_map_opcode.data['row_pitch']
    cp = gain_map_opcode.data['col_pitch']

    gm_w = right - left
    gm_h = bottom - top

    # gain_map = cv2.resize(gain_map, dsize=(gm_w, gm_h), interpolation=cv2.INTER_LINEAR)

    # TODO
    # if top > 0:
    #     pass
    # elif left > 0:
    #     left_col = gain_map[:, 0:1]
    #     rep_left_col = np.tile(left_col, [1, left])
    #     gain_map = np.concatenate([rep_left_col, gain_map], axis=1)
    # elif bottom < raw_image.shape[0]:
    #     pass
    # elif right < raw_image.shape[1]:
    #     pass

    result_image = raw_image.copy()

    # one channel
    # result_image[::rp, ::cp] *= gain_map[::rp, ::cp]

    # per bayer channel
    upper_left_idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    bayer_pattern_idx = np.array(bayer_pattern)
    # blue channel index --> 3
    bayer_pattern_idx[bayer_pattern_idx == 2] = 3
    # second green channel index --> 2
    if bayer_pattern_idx[3] == 1:
        bayer_pattern_idx[3] = 2
    else:
        bayer_pattern_idx[2] = 2
    for c in range(4):
        i0 = upper_left_idx[c][0]
        j0 = upper_left_idx[c][1]
        result_image[i0::2, j0::2] *= gain_map[:, :, bayer_pattern_idx[c]]

    if clip:
        result_image = np.clip(result_image, 0.0, 1.0)

    return result_image


def get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type='VNG'):
    # using opencv edge-aware demosaicing
    if alg_type != '':
        alg_type = '_' + alg_type
    if output_channel_order == 'BGR':
        if cfa_pattern == [0, 1, 1, 2]:  # RGGB
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2BGR' + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2BGR' + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2BGR' + alg_type)
        else:
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
            print("CFA pattern not identified.")
    else:  # RGB
        if cfa_pattern == [0, 1, 1, 2]:  # RGGB
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2RGB' + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2RGB' + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2RGB' + alg_type)
        else:
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
            print("CFA pattern not identified.")
    return opencv_demosaic_flag


def denoise(image, alg_type='fgs'):
    if alg_type == 'fgs':
        guide = (image * 255).astype(dtype=np.uint8)
        denoised_image = cv2.ximgproc.fastGlobalSmootherFilter(guide, image, 100, 0.75)

    else:
        denoised_image = image

    return denoised_image


def demosaic(bayer_image, cfa_pattern, output_channel_order='RGB', alg_type='VNG'):
    """
    Demosaic a Bayer image.
    :param bayer_image: Image in Bayer format, single channel.
    :param cfa_pattern: Bayer/CFA pattern.
    :param output_channel_order: Either RGB or BGR.
    :param alg_type: algorithm type. options: '', 'EA' for edge-aware, 'VNG' for variable number of gradients
    :return: Demosaiced image.
    """
    if alg_type == 'VNG':
        max_val = 255
        wb_image = (bayer_image * max_val).astype(dtype=np.uint8)
    else:
        max_val = 16383
        wb_image = (bayer_image * max_val).astype(dtype=np.uint16)

    if alg_type in ['', 'EA', 'VNG']:
        opencv_demosaic_flag = get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type=alg_type)
        demosaiced_image = cv2.cvtColor(wb_image, opencv_demosaic_flag)
    elif alg_type == 'menon2007':
        cfa_pattern_str = "".join(["RGB"[i] for i in cfa_pattern])
        demosaiced_image = demosaicing_CFA_Bayer_Menon2007(wb_image, pattern=cfa_pattern_str)
    else:
        raise ValueError('Unsupported demosaicing algorithm, alg_type = {}'.format(alg_type))

    demosaiced_image = demosaiced_image.astype(dtype=np.float32) / max_val

    return demosaiced_image

def apply_color_space_transform(image, color_matrix_1, color_matrix_2, illuminant=None, interpolate_csts=True):
    if type(color_matrix_1[0]) is Ratio:
        color_matrix_1 = ratios2floats(color_matrix_1)
    if type(color_matrix_2[0]) is Ratio:
        color_matrix_2 = ratios2floats(color_matrix_2)
    xyz2cam1 = np.reshape(np.asarray(color_matrix_1), (3, 3))
    xyz2cam2 = np.reshape(np.asarray(color_matrix_2), (3, 3))
    # normalize rows (needed?)
    xyz2cam1 = xyz2cam1 / np.sum(xyz2cam1, axis=1, keepdims=True)
    xyz2cam2 = xyz2cam2 / np.sum(xyz2cam1, axis=1, keepdims=True)

    if interpolate_csts and illuminant is not None:
        # interpolate between CSTs based on illuminant
        cct = raw_rgb_to_cct(illuminant, xyz2cam1, xyz2cam2)
        # print(cct)
        xyz2cam_interp = interpolate_cst(xyz2cam1, xyz2cam2, cct)
        xyz2cam_interp = xyz2cam_interp / np.sum(xyz2cam_interp, axis=1, keepdims=True)
        cam2xyz_interp = np.linalg.inv(xyz2cam_interp)
        xyz_image = cam2xyz_interp[np.newaxis, np.newaxis, :, :] * image[:, :, np.newaxis, :]
    else:
        # for now, use one matrix
        # simplified matrix multiplication
        # inverse
        cam2xyz1 = np.linalg.inv(xyz2cam1)
        xyz_image = cam2xyz1[np.newaxis, np.newaxis, :, :] * image[:, :, np.newaxis, :]

    xyz_image = np.sum(xyz_image, axis=-1)
    xyz_image = np.clip(xyz_image, 0.0, 1.0)
    return xyz_image

def transform_xyz_to_srgb(xyz_image):
    # srgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
    #                      [0.2126729, 0.7151522, 0.0721750],
    #                      [0.0193339, 0.1191920, 0.9503041]])

    # xyz2srgb = np.linalg.inv(srgb2xyz)

    xyz2srgb = np.array([[3.2404542, -1.5371385, -0.4985314],
                         [-0.9692660, 1.8760108, 0.0415560],
                         [0.0556434, -0.2040259, 1.0572252]])

    # normalize rows (needed?)
    xyz2srgb = xyz2srgb / np.sum(xyz2srgb, axis=-1, keepdims=True)

    srgb_image = xyz2srgb[np.newaxis, np.newaxis, :, :] * xyz_image[:, :, np.newaxis, :]
    srgb_image = np.sum(srgb_image, axis=-1)
    srgb_image = np.clip(srgb_image, 0.0, 1.0)
    return srgb_image


def fix_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW

    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def reverse_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW
    rev_orientations = np.array([1, 2, 3, 4, 5, 8, 7, 6])
    return fix_orientation(image, rev_orientations[orientation - 1])


def apply_gamma(x):
    return x ** (1.0 / 2.2)


def apply_tone_map(x, tone_curve='simple-s-curve'):
    if tone_curve == 'simple-s-curve':
        tone_mapped_image = 3 * x ** 2 - 2 * x ** 3
    else:
        # tone_curve = loadmat('tone_curve.mat')
        tone_curve = loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tone_curve.mat'))
        tone_curve = tone_curve['tc']
        x = np.round(x * (len(tone_curve) - 1)).astype(int)
        tone_mapped_image = np.squeeze(tone_curve[x])
    return tone_mapped_image


def apply_local_tone_map(x, alg_type='clahe', channel_order='RGB', clahe_clip_limit=1.0, clahe_grid_size=(8, 8)):
    if alg_type == 'clahe':
        to_ycrcb_flag = eval('cv2.COLOR_{}2YCR_CB'.format(channel_order))
        from_ycrcb_flag = eval('cv2.COLOR_YCR_CB2{}'.format(channel_order))

        max_val = 65535
        x = (x * max_val).astype(dtype=np.uint16)

        y_cr_cb = cv2.cvtColor(x, to_ycrcb_flag)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)
        x = np.dstack((clahe.apply(y_cr_cb[:, :, 0]), y_cr_cb[:, :, 1], y_cr_cb[:, :, 2]))
        x = cv2.cvtColor(x, from_ycrcb_flag)

        x = x.astype(dtype=np.float32) / max_val

    else:
        raise ValueError('Unsupported local tone mapping algorithm, alg_type = {}'.format(alg_type))
    return x


# def raw_rgb_to_cct(rawRgb, xyz2cam1, xyz2cam2):
#     """Convert raw-RGB triplet to corresponding correlated color temperature (CCT)"""
#     pass
#     # pxyz = [.5, 1, .5]
#     # loss = 1e10
#     # k = 1
#     # while loss > 1e-4:
#     #     cct = XyzToCct(pxyz)
#     #     xyz = RawRgbToXyz(rawRgb, cct, xyz2cam1, xyz2cam2)
#     #     loss = norm(xyz - pxyz)
#     #     pxyz = xyz
#     #     fprintf('k = %d, loss = %f\n', [k, loss])
#     #     k = k + 1
#     # end
#     # temp = cct


def process_to_save(image, out_dtype='uint8', channel_order='bgr'):
    """
    Process an RGB image to be saved with OpenCV.
    :param image: Input image.
    :param out_dtype: Target data type (e.g., 'uint8', 'uint16', ...).
    :param channel_order: Output channel order (e.g., 'bgr' for OpenCV, ...).
    :return: Processed image in the target data type and channel order.
    """
    in_dtype = str(image.dtype)

    if in_dtype != out_dtype:

        # normalize with source data type
        if in_dtype == 'uint8':
            image = image.astype('float32') / 255.0
        elif in_dtype == 'uint16':
            image = image.astype('float32') / 65535.0
        else:
            pass  # assuming float

        # quantize with target data type
        if out_dtype == 'uint8':
            max_val = 255
        elif out_dtype == 'uint16':
            max_val = 65535
        else:
            max_val = 255  # default
        image = (image * max_val).astype(out_dtype)

    # rearrange channel order, if needed
    if channel_order == 'bgr':
        image = image[:, :, ::-1]

    return image


def fix_missing_params(params):
    """
    Fix params dictionary by filling missing parameters with default values.
    :param params: Input params dictionary.
    :return: Fixed params dictionary.
    """
    params_fixed = params.copy()
    default_params = {
        'input_stage': 'raw',
        'output_stage': 'tone',
        'save_as': 'jpg',
        'white_balancer': 'default',
        'demosaicer': '',
        'denoiser': 'fgs',
        'tone_curve': 'simple-s-curve',
        'local_tone_mapping': 'clahe',  # options: 'clahe', or self-defined module
        'clahe_clip_limit': 1.0,
        'clahe_grid_size': (8, 8)
    }
    for key in default_params.keys():
        if key not in params_fixed:
            params_fixed[key] = default_params[key]
    return params_fixed
