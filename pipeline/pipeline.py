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

"""

from .pipeline_utils import get_visible_raw_image, get_metadata, normalize, white_balance, demosaic, \
    denoise, apply_color_space_transform, transform_xyz_to_srgb, apply_gamma, apply_tone_map, fix_orientation, \
    fix_missing_params, lens_shading_correction, active_area_cropping, lens_distortion_correction, default_cropping, \
    resize, apply_local_tone_map


def run_module(image, module, built_in_function, built_in_args):
    if type(module) is list and len(module) == 2:
        image = module[0](image, **module[1])
    elif type(module) is str:
        image = built_in_function(image, **built_in_args)
    else:
        raise ValueError('Invalid input module.')
    return image


def linearization_stage(current_image, metadata):
    linearization_table = metadata['linearization_table']
    if linearization_table is not None:
        print('Linearization table found. Not handled.')
        # TODO
    return current_image


def lens_shading_correction_stage(current_image, metadata, clip):
    if 'opcode_lists' in metadata:
        if 51009 in metadata['opcode_lists']:
            opcode_list_2 = metadata['opcode_lists'][51009]
            gain_map_opcode = opcode_list_2[9]
            current_image = lens_shading_correction(current_image, gain_map_opcode=gain_map_opcode,
                                                    bayer_pattern=metadata['cfa_pattern'], clip=clip)
    return current_image


def lens_distortion_correction_stage(current_image, metadata, clip):
    if 'opcode_lists' in metadata:
        if 51022 in metadata['opcode_lists']:
            opcode_list_3 = metadata['opcode_lists'][51022]
            rect_warp_opcode = opcode_list_3[1]
            current_image = lens_distortion_correction(current_image, rect_warp_opcode=rect_warp_opcode,
                                                    clip=clip)
    return current_image


def run_pipeline(image_or_path, params=None, metadata=None, stages=None, clip=True):
    if type(image_or_path) == str:
        image_path = image_or_path
        # raw image data
        raw_image = get_visible_raw_image(image_path)
        # metadata
        metadata = get_metadata(image_path)
    else:
        raw_image = image_or_path.copy()
        # must provide metadata
        if metadata is None:
            raise ValueError("Must provide metadata when providing image data in first argument.")

    # take a deep copy of params as it will be modified below
    params = params.copy()

    # fill any missing parameters with default values
    params = fix_missing_params(params)

    '''
    Function performed at each stage. Follows this format:
    * {'stage_name': [function_name, function_params]} 
    * Assumes the function takes in `current_image` as the first parameter. 
    '''
    operation_by_stage = {
        'active_area_cropping': [active_area_cropping, {'active_area': metadata['active_area']}],
        'default_cropping': [default_cropping, {'default_crop_origin': metadata['default_crop_origin'],
                                                'default_crop_size': metadata['default_crop_size']}],
        'linearization': [linearization_stage, {'metadata': metadata}],
        'normal': [normalize, {
            'black_level': metadata['black_level'],
            'white_level': metadata['white_level'],
            'clip': clip}],
        'lens_shading_correction': [lens_shading_correction_stage, {'metadata': metadata, 'clip': clip}],
        'white_balance': [run_module, {
            'module': params['white_balancer'],
            'built_in_function': white_balance,
            'built_in_args': {
                'as_shot_neutral': metadata['as_shot_neutral'],
                'cfa_pattern': metadata['cfa_pattern'],
                'clip': clip
            }
        }],
        'demosaic': [run_module, {
            'module': params['demosaicer'],
            'built_in_function': demosaic,
            'built_in_args': {
                'cfa_pattern': metadata['cfa_pattern'],
                'output_channel_order': 'RGB',
                'alg_type': params['demosaicer']
            }
        }],
        'lens_distortion_correction': [lens_distortion_correction_stage, {'metadata': metadata, 'clip': clip}],
        'denoise': [run_module, {
            'module': params['denoiser'],
            'built_in_function': denoise,
            'built_in_args': {
                'alg_type': params['denoiser']
            }
        }],
        'xyz': [apply_color_space_transform, {
            'color_matrix_1': metadata['color_matrix_1'],
            'color_matrix_2': metadata['color_matrix_2'],
            'illuminant': metadata['as_shot_neutral']
        }],
        'srgb': [transform_xyz_to_srgb, {}],
        'fix_orient': [fix_orientation, {'orientation': metadata['orientation']}],
        'gamma': [apply_gamma, {}],
        'tone': [run_module, {
            'module': params['tone_curve'],
            'built_in_function': apply_tone_map,
            'built_in_args': {
                'tone_curve': params['tone_curve']
            }
        }],
        'resize': [resize, {'target_size': (raw_image.shape[1], raw_image.shape[0])}],
        'local_tone_mapping': [run_module, {
            'module': params['local_tone_mapping'],
            'built_in_function': apply_local_tone_map,
            'built_in_args': {
                'alg_type': params['local_tone_mapping'],
                'channel_order': 'RGB',
                'clahe_clip_limit': params['clahe_clip_limit'],
                'clahe_grid_size': params['clahe_grid_size']
            }
        }],
    }

    if not stages:
        stages = ['raw', 'active_area_cropping', 'linearization', 'normal', 'lens_shading_correction', 'white_balance',
                  'demosaic', 'lens_distortion_correction', 'denoise', 'xyz', 'srgb', 'fix_orient', 'gamma', 'tone',
                  'local_tone_mapping', 'default_cropping', 'resize']

    input_stage = params['input_stage']
    output_stage = params['output_stage']
    if input_stage not in stages \
            or output_stage not in stages \
            or stages.index(input_stage) > stages.index(output_stage):
        raise ValueError('Invalid input/output stage: input_stage = {}, output_stage = {}'.format(input_stage,
                                                                                                  output_stage))
    input_idx = stages.index(input_stage)
    output_idx = stages.index(output_stage)
    current_image = raw_image

    for stage in stages[input_idx+1:output_idx+1]:
        operation = operation_by_stage[stage]
        current_image = operation[0](current_image, **operation[1])
    return current_image
