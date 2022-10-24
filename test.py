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

from utils.unet import UNet
import argparse
import os, time, datetime
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import cv2
import torch
from pipeline.pipeline import run_pipeline
from pipeline.pipeline_utils import get_metadata, get_visible_raw_image
from utils.gen_utils import check_dir


def to_tensor(img):
    img = torch.from_numpy(img.astype(np.float32))
    img = img.unsqueeze(0).permute(0, 3, 1, 2)
    return img


def from_tensor(img):
    img = img.permute(0, 2, 3, 1)
    img = img.cpu().detach().numpy()
    return np.squeeze(img)


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def save_result(result, path):
    path = path if path.find('.') != -1 else path + '.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        cv2.imwrite(path, result)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='dataset/night_real', type=str, help='directory of test dataset')
    parser.add_argument('--set_names', default='iso_1600,iso_3200', type=str, help='name of test dataset')
    parser.add_argument('--model_dir', default='mymodel', type=str, help='directory of the model')
    parser.add_argument('--model_name', default='bestmodel.pt', type=str, help='name of the model')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of results')
    parser.add_argument('--save_result', default=True, type=bool, help='save image')
    args = parser.parse_args()


    stages = ['raw', 'active_area_cropping', 'linearization', 'normal', 'white_balance',
                  'demosaic']

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

    set_names_list = [item for item in args.set_names.split(',')]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=3, init_features=32)
    model.load_state_dict(torch.load(os.path.join('models',args.model_dir, args.model_name)))
    model = model.to(device)
    model.eval()  # Set model to evaluate mode
    print('model loaded')

    for folder_index, set_cur in enumerate(set_names_list):
        print(set_cur)

        fullsavepath = os.path.join(args.result_dir,
                                   os.path.basename(args.set_dir) + '_' + set_cur + '_' + os.path.basename(args.model_dir) + '_' + args.model_name[:-3])
        check_dir(args.result_dir)
        check_dir(fullsavepath)

        psnrs = []
        ssims = []

        fsrgb = open(os.path.join(fullsavepath, "results.txt"), "w")

        for c, im in enumerate(sorted(os.listdir(os.path.join(args.set_dir, 'dng', set_cur)))):
            print('Processing ' + str(c + 1))
            if set_cur == 'iso_50':
                x = cv2.imread(os.path.join(args.set_dir, 'clean_raw', im[:-4]+'.png'), cv2.IMREAD_UNCHANGED)
            else:
                x = get_visible_raw_image(os.path.join(args.set_dir, 'dng', set_cur, im))

            meta_path = os.path.join(args.set_dir, 'dng', set_cur, im)
            meta_data_org = get_metadata(meta_path)

            x = run_pipeline(x, params=params, metadata=meta_data_org, stages=stages)
            x = to_tensor(x)
            x = x ** (1 / 2.2)

            y = cv2.imread(os.path.join(args.set_dir, 'clean', im[:-4]+'.png'), cv2.IMREAD_UNCHANGED)
            y = np.array(cv2.cvtColor(y, cv2.COLOR_BGR2RGB), dtype=np.float32)
            if y.shape[0] > y.shape[1]:
                y = np.rot90(y)
            y = y.astype('uint8')

            x = x.to(device)
            start_time = time.time()
            with torch.no_grad():
                y_ = model(x)  # inference
            elapsed_time = time.time() - start_time
            y_ = y_.cpu()

            y_ = (y_).permute(0, 2, 3, 1).numpy()
            y_ = np.clip(np.squeeze(y_), 0, 1)
            y_ = (255 * y_).astype('uint8')

            if args.save_result:
                name, ext = os.path.splitext(im)
                save_result(y_[:, :, [2, 1, 0]],
                            path=os.path.join(fullsavepath, name + '_output.png'))  # save the result

            psnr_x_ = compare_psnr(y, y_, data_range=255)
            ssim_x_ = compare_ssim(y, y_, multichannel=True, data_range=255)
            psnrs.append(psnr_x_)
            ssims.append(ssim_x_)

            log('{0:10s} \n PSNR = {1:2.2f}dB, SSIM = {2:1.4f}, Time = {3:2.4f} seconds'.format(im, psnr_x_,
                                                                                            ssim_x_, elapsed_time))
            fsrgb.write('{0:10s} : PSNR = {1:2.2f}dB, SSIM = {2:1.4f}, Time = {3:2.4f} seconds \n'.format(im, psnr_x_,
                                                                                                     ssim_x_, elapsed_time))

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)

        print()
        log('Dataset: {0:10s} \n Avg. PSNR = {1:2.4f}dB, Avg. SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))

        fsrgb.write('\nDataset: {0:10s} \n Avg. PSNR = {1:2.4f}dB, Avg. SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))
        fsrgb.close()
