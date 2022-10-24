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


Demo saving the output of each ISP stage (linear raw, demosaiced, ...)
in the same directory as the input image.
"""

import numpy as np
import cv2
from argparse import ArgumentParser
from pipeline.pipeline_utils import get_visible_raw_image, get_metadata, save_image_stage
from .pipeline import run_pipeline


def demo_all_stages(dng_fn):
    stages = ['raw', 'active_area_cropping', 'linearization', 'normal', 'white_balance',
              'demosaic', 'xyz', 'srgb', 'fix_orient', 'gamma', 'tone']

    params = {
        'save_as': 'jpg',               # options: 'jpg', 'png', 'tif', etc.
        'white_balancer': 'default',    # options: default, or self-defined module
        'demosaicer': 'menon2007',      # options: '' for simple interpolation,
                                        #          'EA' for edge-aware,
                                        #          'VNG' for variable number of gradients,
                                        #          'menon2007' for Menon's algorithm
        'tone_curve': 'simple-s-curve',  # options: 'simple-s-curve', 'default', or self-defined module
    }

    # read image and metadata
    image = get_visible_raw_image(dng_fn)
    metadata = get_metadata(dng_fn)

    # save raw stage
    save_image_stage(image.astype(np.float32) / metadata['white_level'], dng_fn, stages[0], params['save_as'])

    for i in range(len(stages) - 1):
        params['input_stage'] = stages[i]
        params['output_stage'] = stages[i + 1]
        image = run_pipeline(image, metadata=metadata, params=params, stages=stages)

        # save current stage
        save_image_stage(image, dng_fn, stages[i + 1], params['save_as'])


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dng_fn', type=str, help='DNG file path.')
    args = arg_parser.parse_args()
    demo_all_stages(args.dng_fn)
