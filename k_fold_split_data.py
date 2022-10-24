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

from shutil import copyfile, rmtree
import os
from glob import glob
import argparse
import cv2
import pickle

from pipeline.pipeline_utils import get_metadata, get_visible_raw_image
from utils.gen_utils import check_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default='dataset/night_real', type=str,
                        help='base address')
    parser.add_argument('--which_fold', default=0, type=int, help='which fold for testing [0,1,2]')
    parser.add_argument('--kfold_indices', default='utils/k_fold_indices_saved.p', type=str, help='load saved k-fold indices')
    parser.add_argument('--with_noise', default=0, type=int, help='for noisy case, mix iso 1600 and 3200 equally')
    args = parser.parse_args()

    print(args)

    return args


def split_func(args, folder_path, k_fold_indices):

    train_index = k_fold_indices['train_index_all'][args.which_fold]
    val_index = k_fold_indices['val_index_all'][args.which_fold]
    test_index = k_fold_indices['test_index_all'][args.which_fold]

    if folder_path == 'iso_1600':
        index = {'train': train_index[0::2], 'val': val_index[0::2]}
    elif folder_path == 'iso_3200':
        index = {'train': train_index[1::2], 'val': val_index[1::2]}
    else:
        index = {'train': train_index, 'val': val_index}

    input_dir = args.base_path

    print('train')
    print(index['train'])
    print('...')

    print('val')
    print(index['val'])
    print('...')

    print('test')
    print(test_index)
    print('...')
    print(args.which_fold, len(index['train']), len(index['val']), len(test_index))

    # create directories if they don't exist
    check_dir('data')
    for fol in ['train', 'val', 'test']:  check_dir(os.path.join('data', fol))
    for fol in ['train', 'val']:
        for subfol in ['clean_raw', 'noisy_raw', 'clean', 'metadata_raw']: check_dir(os.path.join('data', fol, subfol))
    for fol in ['test']:
        for subfol in ['clean_raw', 'clean', 'dng']: check_dir(os.path.join('data', fol, subfol))
    check_dir(os.path.join('data', 'test', 'dng', folder_path))

    allfiles = [os.path.basename(x) for x in sorted(glob(os.path.join(input_dir, 'clean', '*.png')))]

    for fol in ['train', 'val']:
        for ind in index[fol]:
            for subfol in ['clean_raw', 'clean']:
                source = os.path.join(input_dir, subfol, allfiles[ind])
                destination = os.path.join('data', fol, subfol, allfiles[ind])
                copyfile(source, destination)

            for subfol in ['noisy_raw']:
                rawimg = get_visible_raw_image(os.path.join(input_dir, 'dng', folder_path, allfiles[ind][:-4] + '.dng'))
                destination = os.path.join('data', fol, subfol, allfiles[ind])
                cv2.imwrite(destination, rawimg)

            for subfol in ['metadata_raw']:
                metadata = get_metadata(os.path.join(input_dir, 'dng', folder_path, allfiles[ind][:-4] + '.dng'))
                pickle.dump(metadata, open(os.path.join('data', fol, subfol, allfiles[ind][:-4] + '.p'), "wb"))

    for fol in ['test']:
        for ind in test_index:
            if folder_path == 'iso_50' or folder_path == 'iso_1600':
                for subfol in ['clean_raw', 'clean']:
                    source = os.path.join(input_dir, subfol, allfiles[ind])
                    destination = os.path.join('data', fol, subfol, allfiles[ind])
                    copyfile(source, destination)

            for subfol in ['dng']:
                source = os.path.join(input_dir, subfol, folder_path, allfiles[ind][:-4]+'.dng')
                destination = os.path.join('data', fol, subfol, folder_path, allfiles[ind][:-4]+'.dng')
                copyfile(source, destination)


if __name__ == "__main__":
    args = parse_args()
    k_fold_indices = pickle.load(open(args.kfold_indices, "rb"))

    if os.path.isdir('data'): rmtree('data')

    if args.with_noise:
        split_func(args, 'iso_1600', k_fold_indices)
        split_func(args, 'iso_3200', k_fold_indices)
    else:
        split_func(args, 'iso_50', k_fold_indices)

    print('Done!')
