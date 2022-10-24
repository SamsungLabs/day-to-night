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

# no need to run this code separately

import glob
import cv2
import numpy as np
import pickle

from pipeline.pipeline import run_pipeline

aug_times = 1

params = {
    'save_as': 'png',  # options: 'jpg', 'png', 'tif', etc.
    'white_balancer': 'default',  # options: default, or self-defined module
    'demosaicer': '',  # options: '' for simple interpolation,
    #          'EA' for edge-aware,
    #          'VNG' for variable number of gradients,
    #          'menon2007' for Menon's algorithm
    'tone_curve': 'simple-s-curve',  # options: 'simple-s-curve', 'default', or self-defined module
    'output_stage': 'demosaic',
}

stages = ['raw', 'active_area_cropping', 'linearization', 'normal', 'white_balance',
          'demosaic']


def data_aug(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def get_all_patches(img, patch_size=48, stride=48):
    h, w, cc = img.shape
    patches = []
    # extract patches
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            x = img[i:i + patch_size, j:j + patch_size, :]
            # data aug
            for k in range(0, aug_times):
                x_aug = data_aug(x, mode=0)  # np.random.randint(0,8))
                patches.append(x_aug)
    return patches


def gen_patches_sRGB(file_name, patch_size=48, stride=48):
    # read image
    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.float32)
    patches = get_all_patches(img, patch_size, stride)
    return patches


def gen_patches_raw(file_name, meta_name, wb_illum='avg', patch_size=48, stride=48):

    meta_data_org = pickle.load(open(meta_name, "rb"))

    if wb_illum == 'avg':
        meta_data_org['as_shot_neutral'] = meta_data_org['avg_night_illuminant']  # modify as_shot_neutral

    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

    img = run_pipeline(img, params=params, metadata=meta_data_org, stages=stages)

    patches = get_all_patches(img, patch_size, stride)

    return patches


def datagenerator_sRGB(data_dir='dummy_dataset/train/clean', batch_size=128, patch_size=48, stride=48, verbose=False):
    file_list = sorted(glob.glob(data_dir + '/*.png'))  # get name list of all .png files
    # initialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patch = gen_patches_sRGB(file_list[i], patch_size=patch_size, stride=stride)
        data.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done')
    data = np.concatenate(data)
    discard_n = len(data) - len(data) // batch_size * batch_size
    data = np.delete(data, range(discard_n), axis=0)
    # print('Number of patches ' + str(len(data[0])))
    print('^_^-sRGB training data finished-^_^')
    return data


def datagenerator_raw(data_dir='dummy_dataset/train/clean_raw', meta_dir='dummy_dataset/train/metadata_raw', wb_illum='avg',
                  batch_size=128, patch_size=48, stride=48, verbose=False):

    file_list = sorted(glob.glob(data_dir + '/*.png'))  # get name list of all .png files

    meta_list = sorted(glob.glob(meta_dir + '/*.p'))  # get name list of all .p files

    # initialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patch = gen_patches_raw(file_list[i], meta_list[i], wb_illum=wb_illum, patch_size=patch_size, stride=stride)
        data.append(patch)
        if verbose:
            print(str(i + 1) + '/' + str(len(file_list)) + ' is done')
    data = np.concatenate(data)
    discard_n = len(data) - len(data) // batch_size * batch_size
    data = np.delete(data, range(discard_n), axis=0)
    # print('Number of patches ' + str(len(data[0])))
    print('^_^-raw training data finished-^_^')
    return data


if __name__ == '__main__':
    batch_size, patch_size, stride = 128, 48, 48
    data_sRGB = datagenerator_sRGB(data_dir='real_night/train/clean', batch_size=batch_size, patch_size=patch_size,
                         stride=stride)
    data_raw = datagenerator_raw(data_dir='real_night/train/clean_raw', meta_dir='dummy_dataset/train/metadata_raw',
                                 wb_illum='avg', batch_size=batch_size, patch_size=patch_size, stride=stride)
    print('Done!')
