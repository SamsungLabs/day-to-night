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

"""

from shutil import copyfile
import os
from glob import glob
import numpy as np
import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default='data', type=str,
                        help='base address of real data')
    parser.add_argument('--synth_path',
                        default='synthetic_datasets/night',
                        type=str,
                        help='synth data path')
    parser.add_argument('--real_percent',
                        default='10',
                        type=int,
                        help='percentage of real images, rest will be synthetic')
    parser.add_argument('--total_train',
                        default='60',
                        type=int,
                        help='percentage of real images, rest will be synthetic')
    parser.add_argument(
        '--seed', default=101, type=int, help='seed np random generator')
    args = parser.parse_args()

    print(args)

    return args


def remove_data(remove_ind, args):
    input_dir = os.path.join(args.base_path, 'train')
    allfiles = [os.path.basename(x) for x in sorted(glob(os.path.join(input_dir, 'clean', '*.png')))]

    for i in remove_ind:
        for fol in ['clean_raw', 'noisy_raw', 'clean']:
            os.remove(os.path.join(input_dir, fol, allfiles[i]))

        for fol in ['metadata_raw']:
            os.remove(os.path.join(input_dir, fol, allfiles[i][:-4] + '.p'))


def add_data(add_ind, args):
    input_dir = os.path.join(args.base_path, 'train')
    allfilessynth = [os.path.basename(x) for x in sorted(glob(os.path.join(args.synth_path, 'train', 'clean', '*.png')))]

    for i in add_ind:
        for fol in ['clean_raw', 'clean', 'noisy_raw']:
            source = os.path.join(args.synth_path, 'train', fol, allfilessynth[i])
            destination = os.path.join(input_dir, fol, allfilessynth[i])
            copyfile(source, destination)

        for fol in ['metadata_raw']:
            source = os.path.join(args.synth_path, 'train', fol, allfilessynth[i][:-4] + '.p')
            metadata = pickle.load(open(source, "rb"))
            metadata['as_shot_neutral'] = metadata['avg_night_illuminant']
            destination = os.path.join(input_dir, fol, allfilessynth[i][:-4] + '.p')
            pickle.dump(metadata, open(destination, "wb"))


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    randperm = np.random.permutation(args.total_train)

    # number of real images
    num_real = round(args.real_percent * args.total_train / 100)
    # remaining real images will be removed
    # and replaced with synthetic images
    remove_real = args.total_train - num_real

    # remove remaining real images
    ind = randperm[0:remove_real]
    remove_data(ind, args)

    # replace removed with synthetic data
    add_data(ind, args)

    print('Done!')
