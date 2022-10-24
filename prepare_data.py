"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author(s):
Abhijith Punnappurath (abhijith.p@samsung.com)
Abdullah Abuolaim (abdullah.abuolaim@gmail.com)
Abdelrahman Abdelhamed (a.abdelhamed@samsung.com; abdoukamel@gmail.com)


Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

Description:
Generating synthetic night images from day images.
"""

import pickle
import numpy as np
import cv2
import os
import random
import argparse
from glob import glob
import scipy.io
from copy import deepcopy

from utils.relight import relight_locally, apply_local_lights_rgb
from pipeline.pipeline import run_pipeline
from pipeline.pipeline_utils import normalize, denormalize, get_visible_raw_image, ratios2floats, white_balance, \
    get_metadata
from utils.gen_utils import check_dir

from noise_profiler.image_synthesizer import load_noise_model, synthesize_noisy_image_v2
noise_model_path = './noise_profiler/h-gauss-s20-v1'
noise_model, iso2b1_interp_splines, iso2b2_interp_splines = load_noise_model(path=noise_model_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_address', type=str,
                        help='path to day dataset',
                        default='./dataset/day/'
                        )
    parser.add_argument('--savefolderpath', default='synthetic_datasets', type=str, help='main path to save to')
    parser.add_argument('--savefoldername', default='night', type=str, help='directory to save to')
    parser.add_argument('--how_many_train', default=60, type=int, help='how many training images')

    parser.add_argument('--dim', default=False, action='store_true', help='dim or not')
    parser.add_argument('--relight', default=False, action='store_true', help='relight or not')
    parser.add_argument('--discard_black_level', default=False, type=bool,
                        help='whether to discard black-level subtraction or not')
    parser.add_argument('--clip', default=True, type=bool, help='whether to clip or not')

    parser.add_argument('--relight_local', default=False, action='store_true', help='locally relight or not')
    parser.add_argument('--min_num_lights', default=5, type=int, help='min. number of local lights')
    parser.add_argument('--max_num_lights', default=5, type=int, help='max. number of local lights')
    parser.add_argument('--min_light_size', default=.5, type=int, help='min. local light size as percent of image dim.')
    parser.add_argument('--max_light_size', default=1., type=int, help='max. local light size as percent of image dim.')
    parser.add_argument('--save_light_masks', default=False, action='store_true', help='whether to save light masks, '
                                                                                        'disable for speedup')
    parser.add_argument('--num_sat_lights', default=5, type=int, help='number of small saturated local lights')
    parser.add_argument('--iso_list', default='1600,3200', type=str,
                        help='list of discrete ISOs to pick from when adding noise, separate with comma')

    args = parser.parse_args()

    print(args)

    return args


def get_illum_normalized_by_g(illum_in_arr):
    return illum_in_arr[:, 0] / illum_in_arr[:, 1], illum_in_arr[:, 1] / illum_in_arr[:, 1], illum_in_arr[:, 2] / illum_in_arr[:, 1]


def synth_night_imgs(in_img_or_path, in_day_meta_data, dim=True, relight=True, iso=50,
                     discard_black_level=False, _clip=True, relight_local=True,
                     min_num_lights=5, max_num_lights=5, min_light_size = 0.5, max_light_size = 1.0, num_sat_lights=5):

    """
    Synthesizing nigh time images
    :param in_img_or_path:
    :param in_day_meta_data:
    :param dim:
    :param relight:
    :param iso:
    :param discard_black_level:
    :param _clip:
    :param relight_local: Whether to locally relight image.
    :param min_num_lights: Minimum number of local illuminants, in case of local relighting.
    :param max_num_lights: Maximum number of local illuminants, in case of local relighting.
    :param min_light_size: Minimum local light size as percent of image dimension, in case of local relighting.
    :param max_light_size: Maximum local light size as percent of image dimension, in case of local relighting.
    :param num_sat_lights: Number of small saturated local lights, in case of local relighting.
    Return synthetic day-to-night image.
    """
    if type(in_img_or_path) == str:
        image_path = in_img_or_path
        # raw image data
        in_raw_image = get_visible_raw_image(image_path)
    else:
        in_raw_image = in_img_or_path

    meta_data_night = deepcopy(in_day_meta_data)

    if dim:
        sampled_bright = random.uniform(0.55, 0.9)
    else:
        sampled_bright = 1.0

    # relighting
    if relight_local:
        # local illuminants
        num_of_samples = np.random.randint(min_num_lights + num_sat_lights, max_num_lights + num_sat_lights + 1, 1)
        if discard_black_level:
            sampled_wb = (np.random.multivariate_normal(gt_illum_mean, gt_illum_cov, num_of_samples) * (
                    1023 - 64) + 64) / 1023
        else:
            sampled_wb = np.random.multivariate_normal(gt_illum_mean, gt_illum_cov, num_of_samples)

    elif relight:
        num_of_samples = 1
        if discard_black_level:
            sampled_wb = (np.random.multivariate_normal(gt_illum_mean, gt_illum_cov, num_of_samples)[0] * (
                    1023 - 64) + 64) / 1023
        else:
            sampled_wb = np.random.multivariate_normal(gt_illum_mean, gt_illum_cov, num_of_samples)[0]

    else:
        sampled_wb = ratios2floats(in_day_meta_data['as_shot_neutral'])

    if discard_black_level:
        in_raw_image = normalize(in_raw_image, 0, in_day_meta_data['white_level'], clip=_clip)
    else:
        in_raw_image = normalize(in_raw_image, in_day_meta_data['black_level'], in_day_meta_data['white_level'],
                                 clip=_clip)
    white_balanceed_img = white_balance(in_raw_image, ratios2floats(in_day_meta_data['as_shot_neutral']),
                                        in_day_meta_data['cfa_pattern'], clip=_clip)

    if relight_local:
        day_illum = ratios2floats(in_day_meta_data['as_shot_neutral'])
        meta_data_night['as_shot_neutral'] = day_illum
        meta_data_night['day_illuminant'] = day_illum
        meta_data_night['avg_night_illuminant'] = np.mean(sampled_wb, axis=0)
    else:
        meta_data_night['as_shot_neutral'] = sampled_wb

    meta_data_night['iso'] = iso  # sampled_iso

    white_balanceed_img_dim = white_balanceed_img * sampled_bright

    sampled_wb_inv = list(1 / np.asarray(sampled_wb))

    # white imbalance

    if relight_local:
        white_imbalanceed_img_dim_night, local_lights_ = relight_locally(white_balanceed_img_dim, sampled_wb,
                                                                         in_day_meta_data['cfa_pattern'], clip=_clip,
                                                                         invert_wb=True,
                                                                         min_light_size=min_light_size,
                                                                         max_light_size=max_light_size,
                                                                         num_sat_lights=num_sat_lights)
    else:
        white_imbalanceed_img_dim_night = white_balance(white_balanceed_img_dim, sampled_wb_inv,
                                                        in_day_meta_data['cfa_pattern'], clip=_clip)
        local_lights_ = None

    if discard_black_level:
        white_imbalanceed_img_dim_night = (
            denormalize(white_imbalanceed_img_dim_night, 0, meta_data_night['white_level'], clip=_clip)).astype(
            np.uint16)
    else:
        white_imbalanceed_img_dim_night = (
            denormalize(white_imbalanceed_img_dim_night, meta_data_night['black_level'], meta_data_night['white_level'],
                        clip=_clip)).astype(np.uint16)

    if relight_local:
        return white_imbalanceed_img_dim_night, meta_data_night, local_lights_
    else:
        return white_imbalanceed_img_dim_night, meta_data_night


if __name__ == "__main__":

    args = parse_args()

    # load nighttime illuminants
    gt_illum = scipy.io.loadmat('utils/gray_card_illum_dict.mat')
    gt_illum = gt_illum['night_dict']

    gt_illum[:, 0], gt_illum[:, 1], gt_illum[:, 2] = get_illum_normalized_by_g(gt_illum)

    gt_illum_mean = np.mean(gt_illum, 0)
    gt_illum_cov = np.cov(np.transpose(gt_illum))

    # read day images
    base_address = args.base_address
    outdoor_daytime_img_names = [os.path.basename(x) for x in sorted(glob(os.path.join(base_address, '*.dng')))]

    # create save directories
    savefolder = os.path.join(args.savefolderpath, args.savefoldername)
    check_dir(args.savefolderpath)
    check_dir(savefolder)
    for fol in ['train', 'val']:  check_dir(os.path.join(savefolder, fol))
    for fol in ['train', 'val']:
        for subfol in ['clean_raw', 'noisy_raw', 'clean', 'noisy', 'metadata_raw']:
            check_dir(os.path.join(savefolder, fol, subfol))

    if args.relight_local:
        light_mask_dirname = 'masks'
    else:
        light_mask_dirname = None

    if args.save_light_masks and args.relight_local:
        for fol in ['train', 'val']:  check_dir(os.path.join(savefolder, fol, light_mask_dirname))

    # camera pipeline parameters
    stages = ['raw', 'active_area_cropping', 'linearization', 'normal', 'white_balance',
              'demosaic', 'xyz', 'srgb', 'fix_orient', 'gamma', 'tone']

    params = {
        'save_as': 'png',  # options: 'jpg', 'png', 'tif', etc.
        'white_balancer': 'default',  # options: default, or self-defined module
        'demosaicer': 'menon2007',  # options: '' for simple interpolation,
        #          'EA' for edge-aware,
        #          'VNG' for variable number of gradients,
        #          'menon2007' for Menon's algorithm
        'tone_curve': 'simple-s-curve',  # options: 'simple-s-curve', 'default', or self-defined module
    }

    iso_list = [int(item) for item in args.iso_list.split(',')]

    # main loop
    for example_num in range(len(outdoor_daytime_img_names)):

        print('Processing image:', example_num + 1, 'of', len(outdoor_daytime_img_names))

        example_img_bayer_org = get_visible_raw_image(os.path.join(base_address, outdoor_daytime_img_names[example_num]))
        meta_data_org = get_metadata(os.path.join(base_address, outdoor_daytime_img_names[example_num]))

        rand_iso = iso_list[np.random.randint(2)]
        results_ = synth_night_imgs(example_img_bayer_org, meta_data_org, dim=args.dim,
                                    relight=args.relight, iso=rand_iso,
                                    discard_black_level=args.discard_black_level,
                                    _clip=args.clip, relight_local=args.relight_local,
                                    min_num_lights=args.min_num_lights,
                                    max_num_lights=args.max_num_lights,
                                    min_light_size=args.min_light_size,
                                    max_light_size=args.max_light_size,
                                    num_sat_lights=args.num_sat_lights
                                    )

        if args.relight_local:
            example_night_synth, meta_data_night, local_lights = results_
        else:
            example_night_synth, meta_data_night = results_
            local_lights = None

        if example_num >= args.how_many_train:
            curfolder = os.path.join(savefolder, 'val')
        else:
            curfolder = os.path.join(savefolder, 'train')

        cv2.imwrite(curfolder + '/clean_raw/' + outdoor_daytime_img_names[example_num][:-4] + '.png',
                    example_night_synth.astype(np.uint16))
        pickle.dump(meta_data_night,
                    open(curfolder + '/metadata_raw/' + outdoor_daytime_img_names[example_num][:-4] + '.p', "wb"))

        as_shot_neutral = meta_data_night['as_shot_neutral']  # keep as_shot_neutral
        if args.relight_local:
            meta_data_night['as_shot_neutral'] = meta_data_night['avg_night_illuminant']  # modify as_shot_neutral
        night_synth_srgb_avg = run_pipeline(example_night_synth, params=params, metadata=meta_data_night, stages=stages)
        night_synth_srgb_avg = (night_synth_srgb_avg * 255).astype(np.uint8)
        cv2.imwrite(curfolder + '/clean/' + outdoor_daytime_img_names[example_num][:-4] + '.png', night_synth_srgb_avg[:, :, [2, 1, 0]])
        meta_data_night['as_shot_neutral'] = as_shot_neutral  # restore as_shot_neutral

        noisy_night_image = synthesize_noisy_image_v2(example_night_synth, model=noise_model,
                                        dst_iso=meta_data_night['iso'], min_val=0,
                                        max_val=1023,
                                        iso2b1_interp_splines=iso2b1_interp_splines,
                                        iso2b2_interp_splines=iso2b2_interp_splines)

        noisy_night_image = noisy_night_image.astype(np.uint16)

        as_shot_neutral = meta_data_night['as_shot_neutral']  # keep as_shot_neutral
        if args.relight_local:
            meta_data_night['as_shot_neutral'] = meta_data_night['avg_night_illuminant']  # modify as_shot_neutral
        noisy_night_image_srgb = run_pipeline(noisy_night_image, params=params, metadata=meta_data_night, stages=stages)
        noisy_night_image_srgb = (noisy_night_image_srgb * 255).astype(np.uint8)
        meta_data_night['as_shot_neutral'] = as_shot_neutral  # restore as_shot_neutral

        cv2.imwrite(curfolder + '/noisy/' + outdoor_daytime_img_names[example_num][:-4] + '_iso_' + str(rand_iso).zfill(4) + '.png', noisy_night_image_srgb[:, :, [2, 1, 0]])
        cv2.imwrite(curfolder + '/noisy_raw/' + outdoor_daytime_img_names[example_num][:-4] + '.png',
                    noisy_night_image)

        # save local light masks

        if args.relight_local and args.save_light_masks and local_lights is not None:

            neutral_image = np.ones(night_synth_srgb_avg.shape, dtype=np.float32)

            # save individual local light masks
            for k, light in enumerate(local_lights[1:]):
                neutral_image_relight = apply_local_lights_rgb(neutral_image, [local_lights[0], light], clip=True,
                                                               invert_wb=True)
                image_name = '{}_mask_{}_loc_{}_{}.jpg'.format(outdoor_daytime_img_names[example_num][:-4], k,
                                                               light.location[0], light.location[1])
                cv2.imwrite(os.path.join(curfolder, light_mask_dirname, image_name),
                            (neutral_image_relight ** 0.4545 * 255).astype(np.uint8))

            # save combined light mask
            neutral_image_relight = apply_local_lights_rgb(neutral_image, local_lights, clip=True, invert_wb=True)
            image_name = '{}_mask_combined.jpg'.format(outdoor_daytime_img_names[example_num][:-4])
            cv2.imwrite(os.path.join(curfolder, light_mask_dirname, image_name),
                        (neutral_image_relight ** 0.4545 * 255).astype(np.uint8))

        print('Done!')
