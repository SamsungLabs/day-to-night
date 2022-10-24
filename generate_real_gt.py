"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Author:
Abhijith Punnappurath (abhijith.p@samsung.com)

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

Description:
Generating nighttime ground truth by averaging ISO 50 bursts.
Also save noisy inputs.
"""

from shutil import copyfile
from glob import glob
import argparse
import pickle

from pipeline.pipeline_utils import *
from pipeline.pipeline import *
from pipeline.raw_utils import *
from utils.gen_utils import check_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default='full_burst_dataset', type=str,
                        help='base address')
    parser.add_argument('--folder_path', default='2021_09_13_22_45_55_159', type=str, help='burst directory')
    parser.add_argument('--savefolderpath', default='dataset', type=str, help='base path to save to')
    parser.add_argument('--savefoldername', default='night_real', type=str, help='directory to save to')
    parser.add_argument('--iso_list', default='1600,3200', type=str,
                        help='list of discrete ISOs, separate with comma')
    parser.add_argument('--set_of_bursts', default=True, type=bool, help='False: run only on the single burst in folder_path, True: run for all bursts in base_path')

    args = parser.parse_args()

    print(args)

    return args


# camera pipeline stages and parameters
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


def render_direct_average(input_dir, output_dir, iso_list, fol_num):

    # ground truth
    dnglist = sorted(glob(os.path.join(input_dir, 'RAW_0_000050*')))

    first_bayer = get_visible_raw_image(dnglist[0])
    metadata = get_metadata(dnglist[0])
    bayer_avg = np.zeros_like(first_bayer)

    # copy first iso 50 dng
    destination = os.path.join(output_dir, 'dng', 'iso_50', fol_num + '.dng')
    copyfile(dnglist[0], destination)

    for dng in dnglist:
        bayer = get_visible_raw_image(dng)
        bayer_avg += bayer
    bayer_avg = bayer_avg.astype(np.float32)/len(dnglist)

    bayer_avg = bayer_avg.astype(np.uint16)

    avg_img = run_pipeline(bayer_avg, params=params, metadata=metadata, stages=stages)
    avg_img = (avg_img * 255).astype(np.uint8)

    # save clean sRGB
    cv2.imwrite(os.path.join(output_dir, 'clean', fol_num + '.png'),
                avg_img[:, :, [2, 1, 0]])

    # save clean raw
    cv2.imwrite(os.path.join(output_dir, 'clean_raw', fol_num + '.png'),
                bayer_avg)

    for iso in iso_list:
        # copy first dng
        dnginplist = sorted(glob(os.path.join(input_dir, 'RAW_0_00'+str(iso).zfill(4)+'*')))
        destination = os.path.join(output_dir, 'dng', 'iso_'+str(iso), fol_num + '.dng')
        copyfile(dnginplist[0], destination)

    return


if __name__ == "__main__":
    args = parse_args()
    output_dir = os.path.join(args.savefolderpath, args.savefoldername)

    iso_list = [int(item) for item in args.iso_list.split(',')]

    # create directories if they don't exist
    check_dir(args.savefolderpath)
    check_dir(output_dir)
    check_dir(os.path.join(output_dir, 'clean_raw'))
    check_dir(os.path.join(output_dir, 'clean'))
    check_dir(os.path.join(output_dir, 'dng'))
    check_dir(os.path.join(output_dir, 'dng', 'iso_50'))
    for iso in iso_list: check_dir(os.path.join(output_dir, 'dng', 'iso_'+str(iso)))

    if args.set_of_bursts:

        folderlist = [name for name in sorted(os.listdir(args.base_path)) if os.path.isdir(os.path.join(args.base_path, name))]

        # to be consistent with how the experiments were done for the paper
        folder_shuffle_indices = pickle.load(open('./utils/fold_shuffle_indices.p', "rb"))

        for c, fol in enumerate(folderlist):
            input_dir_ = os.path.join(args.base_path, fol)
            print('Processing ' + str(c+1) + ' of ' + str(len(folderlist)))
            print(input_dir_)
            render_direct_average(input_dir_, output_dir, iso_list, str(folder_shuffle_indices[c]).zfill(3))
    else:
        input_dir = os.path.join(args.base_path, args.folder_path)
        print('Processing ' + input_dir)
        fold_num = 999
        render_direct_average(input_dir, output_dir, iso_list, str(fold_num))

    print('Done!')
