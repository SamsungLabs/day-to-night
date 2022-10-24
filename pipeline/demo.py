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


Demo saving the output of the last ISP stage
in the same directory as the input image.
"""

from argparse import ArgumentParser
from pipeline.pipeline_utils import get_visible_raw_image, get_metadata, save_image_stage
from .pipeline import run_pipeline


def demo(dng_fn):
    stages = ['raw', 'active_area_cropping', 'linearization', 'normal', 'white_balance',
              'demosaic', 'xyz', 'srgb', 'fix_orient', 'gamma', 'tone']

    params = {
        'input_stage': 'raw',
        'output_stage': 'tone',
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
    image = run_pipeline(image, metadata=metadata, params=params, stages=stages)

    # save output stage
    save_image_stage(image, dng_fn, params['output_stage'], params['save_as'])


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dng_fn', type=str, help='DNG file path.')
    args = arg_parser.parse_args()
    demo(args.dng_fn)
