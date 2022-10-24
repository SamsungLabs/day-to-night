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

import os
import torch
import torchvision
from torchvision import transforms
import utils.data_generator as dg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DatasetRAW(object):
    def __init__(self, root, batch_size, patch_size, stride, wb_illum, on_cuda, which_input):

        print('Inside raw data generator')
        rawimgs = dg.datagenerator_raw(data_dir=os.path.join(root, which_input),
                                                    meta_dir=os.path.join(root, 'metadata_raw'),
                                                    wb_illum=wb_illum,
                                                    batch_size=batch_size, patch_size=patch_size, stride=stride)
        rawimgs = torch.from_numpy(rawimgs)
        rawimgs = rawimgs.permute(0, 3, 1, 2)

        rawimgs = rawimgs ** (1 / 2.2)
        if on_cuda:
            rawimgs = rawimgs.to(device)

        print('Number of patches '+str(rawimgs.shape[0]))
        print('Outside raw data generator \n')
        self.rawimgs = rawimgs

        print('Inside sRGB data generator')
        srgbimgs = dg.datagenerator_sRGB(data_dir=os.path.join(root, 'clean'), batch_size=batch_size, patch_size=patch_size,
                                    stride=stride)
        print('Number of patches '+str(srgbimgs.shape[0]))
        print('Outside sRGB data generator \n')
        srgbimgs = torch.from_numpy(srgbimgs)
        srgbimgs = srgbimgs.permute(0, 3, 1, 2)
        srgbimgs = srgbimgs / 255.0
        if on_cuda:
            srgbimgs = srgbimgs.to(device)
        self.srgbimgs = srgbimgs

    def __getitem__(self, idx):
        # load images

        img = self.rawimgs[idx]
        target = self.srgbimgs[idx]

        return img, target

    def __len__(self):
        return len(self.rawimgs)


if __name__ == '__main__':
    data_dir = 'synth_night'
    batch_size, patch_size, stride = 4, 480, 480
    wb_illum = 'avg'
    on_cuda = False
    which_input = 'noisy_raw'

    image_datasets = {
        x: DatasetRAW(os.path.join(data_dir, x), batch_size, patch_size, stride, wb_illum,
                      on_cuda, which_input)
        for x in ['val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
                   for x in ['val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}

    # Get a batch of training data
    inputs, targets = next(iter(dataloaders['val']))

    # Make a grid from batch
    targets_grid = torchvision.utils.make_grid(targets)
    inputs_grid = torchvision.utils.make_grid(inputs)

    targets_grid = transforms.ToPILImage()(targets_grid.cpu())
    inputs_grid = transforms.ToPILImage()(inputs_grid.cpu())

    targets_grid.save("debug_targets_grid.png")
    inputs_grid.save("debug_inputs_grid.png")

    print('Done!')
