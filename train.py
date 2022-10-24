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
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from utils.dataset_raw import DatasetRAW
from torch.optim import lr_scheduler
import time
import os
import argparse

from torch.utils.tensorboard import SummaryWriter


def parse_args():

    parser = argparse.ArgumentParser(description='Day-to-night train')
    parser.add_argument(
        '--data-dir', default='data', type=str, help='folder of training and validation images')
    parser.add_argument(
        '--savefoldername', default='models', type=str, help='folder to save trained models to')
    parser.add_argument(
        '--which-input', default='clean_raw', type=str, help='clean_raw or noisy_raw')
    parser.add_argument(
        '--wb-illum', default='asn', type=str, help='asn (as-shot-neutral) or avg (average)')
    parser.add_argument(
        '--patch-size', type=int, default=64, help='patch size')
    parser.add_argument(
        '--stride', type=int, default=64, help='stride when cropping patches')
    parser.add_argument(
        '--batch-size', type=int, default=128, help='batch size')
    parser.add_argument(
        '--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument(
        '--milestones', default='400', type=str, help='milestones as comma separated string')
    parser.add_argument(
        '--num-epochs', type=int, default=500, help='number of epochs')
    parser.add_argument(
        '--tboard-freq', type=int, default=200, help='frequency of writing to tensorboard')
    parser.add_argument('--on-cuda', default=False, action='store_true', help='False: load each batch on cuda, True: load all data directly on cuda')

    args = parser.parse_args()

    print(args)

    return args


def mypsnr(img1, img2):
    mse = torch.mean(((img1 * 255.0).floor() - (img2 * 255.0).floor()) ** 2, dim=[1, 2, 3])
    #    mse[(mse==0).nonzero()]=0.05  # deprecated warning after pytorch 1.5
    mse[torch.nonzero((mse == 0), as_tuple=True)] = 0.05
    psnrout = torch.mean(20 * torch.log10(255.0 / torch.sqrt(mse)))
    return psnrout


def main(args):

    milestones = [item for item in args.milestones.split(',')]
    for i in range(len(milestones)):
        milestones[i] = int(milestones[i])

    savefoldername = args.savefoldername
    writer = SummaryWriter(os.path.join('./tensorboard', savefoldername))
    modsavepath = os.path.join('./models/', savefoldername)
    if not (os.path.exists(modsavepath) and os.path.isdir(modsavepath)):
        os.makedirs(modsavepath)

    image_datasets = {
            x: DatasetRAW(os.path.join(args.data_dir, x), args.batch_size, args.patch_size, args.stride, args.wb_illum,
                          args.on_cuda, args.which_input)
            for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    epoch_loss = {x: 0.0 for x in ['train', 'val']}
    epoch_psnr = {x: 0.0 for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=3, init_features=32)
    model = model.to(device)

    criterion = nn.L1Loss()

    params = model.parameters()
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # training loop starts here
    since = time.time()

    best_loss = 10 ** 6
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        running_loss_tboard = 0.0
        running_psnr_tboard = 0.0

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_psnr = 0.0

            # Iterate over data.
            for i, (inputs, targets) in enumerate(dataloaders[phase]):

                if not args.on_cuda:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    # psnrout = mypsnr((torch.clip(outputs, 0, 1)), targets)
                    psnrout = mypsnr(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        running_loss_tboard += loss.item()
                        running_psnr_tboard += psnrout.item()
                        if (i+1) % args.tboard_freq == 0:  # every tboard_freq mini-batches...

                            # ...log the running loss
                            writer.add_scalar('loss',
                                              running_loss_tboard / args.tboard_freq,
                                              epoch * len(dataloaders[phase]) + i)

                            writer.add_scalar('psnr',
                                              running_psnr_tboard / args.tboard_freq,
                                              epoch * len(dataloaders[phase]) + i)

                            running_loss_tboard = 0.0
                            running_psnr_tboard = 0.0

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_psnr += psnrout.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss[phase] = running_loss / dataset_sizes[phase]
            epoch_psnr[phase] = running_psnr / dataset_sizes[phase]

            if phase == 'val':
                # ...log the running loss
                writer.add_scalars('loss',
                                   {'train': epoch_loss['train'], 'val': epoch_loss['val']},
                                   (epoch + 1) * len(dataloaders['train']))

                writer.add_scalars('psnr',
                                   {'train': epoch_psnr['train'], 'val': epoch_psnr['val']},
                                   (epoch + 1) * len(dataloaders['train']))
                # img_grid = torchvision.utils.make_grid(torch.cat((inputs, torch.clip(outputs, 0, 1),
                #                                                   targets), 2), normalize=True, range=(0, 1))
                img_grid = torchvision.utils.make_grid(torch.cat((inputs, outputs,
                                                                  targets), 2), normalize=True, range=(0, 1))

                # write to tensorboard
                writer.add_image('val_epoch_' + str(epoch), img_grid)

            print('{} Loss: {:.6f} PSNR: {:.4f}'.format(
                phase, epoch_loss[phase], epoch_psnr[phase]))

            # save the model
            if phase == 'val' and epoch_loss[phase] < best_loss:
                best_loss = epoch_loss[phase]
                best_psnr = epoch_psnr[phase]
                torch.save(model.state_dict(), os.path.join(modsavepath, 'bestmodel.pt'))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    print('Best val psnr: {:4f}'.format(best_psnr))


if __name__ == "__main__":

    args = parse_args()
    main(args)
