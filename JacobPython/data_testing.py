import os
import pandas as pd
import xarray as xr
import numpy as np
from DLWP.data import ERA5Reanalysis
from DLWP.model import Preprocessor
from DLWP.model import DLWPFunctional
from DLWP.model.preprocessing import prepare_data_array
from DLWP.model import ArrayDataGenerator
from DLWP.model import tf_data_generator
from tensorflow.keras.layers import Input, UpSampling3D, AveragePooling3D, concatenate, ReLU, Reshape, Concatenate, \
    Permute
from DLWP.custom import CubeSpherePadding2D, CubeSphereConv2D

def check_dataset_variables(file_path):
    try:
        with xr.open_dataset(file_path, engine='netcdf4') as ds:  # Specify the engine explicitly
            print("Variables available in the dataset:")
            print(ds.variables)
    except ValueError as e:
        print(f"Failed to open the dataset at {file_path} with error: {e}")

def main():
    # Configuration
    data_directory = '/Users/jacobholloway/Developer/TestData/'
    processed_file_path = os.path.join(data_directory, 'tutorial_z500_t2m.nc')
    variables = ['z', 't2m']  # Adjusted for actual variable names
    levels = [500, 0]  # Match variables to levels pair-wise
    root_directory = '/Users/jacobholloway/Developer/TestData'
    predictor_file = os.path.join(root_directory, 'tutorial_z500_t2m.nc')
    model_file = os.path.join(root_directory, 'dlwp-cs_tutorial')
    log_directory = os.path.join(root_directory, 'logs', 'dlwp-cs_tutorial')

    # Ensure the output directory exists
    os.makedirs(data_directory, exist_ok=True)
    
    # Initialize ERA5 Reanalysis with custom parameters
    era = ERA5Reanalysis(root_directory=data_directory, file_id='tutorial')
    era.set_variables(variables)
    era.set_levels([l for l in levels if l != 0])  # Exclude single-level from levels for ERA5 setup

    # Initialize the Preprocessor
    pp = Preprocessor(era, predictor_file=processed_file_path)

    
    
    # Process the data into a series format suitable for DLWP model
    pp.data_to_series(
        batch_samples=10000,
        variables=variables,
        levels=levels,
        pairwise=True,
        scale_variables=True,
        overwrite=True,
        verbose=True
    )
    
    # Drop 'varlev' coordinate after processing and save the processed data
    processed_data = pp.data.drop_vars('varlev')  # Updated method call
    processed_data.to_netcdf(processed_file_path + '_nocoord.nc')  # Save to new file
    print(f"Data processed and saved to {processed_file_path}_nocoord.nc")
    
    # Optionally, print data for verification
    print(processed_data)

    # Close resources
    era.close()
    pp.close()

    cnn_model_name = 'unet2'
    base_filter_number = 32
    min_epochs = 0
    max_epochs = 20
    patience = 2
    batch_size = 32
    shuffle = True
    io_selection = {'varlev': ['z/500', 't2m/0']}
    add_solar = False
    io_time_steps = 2
    integration_steps = 2
    data_interval = 2
    loss_by_step = None


    train_set = list(pd.date_range('2013-01-01', '2014-12-31 21:00', freq='3h'))
    validation_set = list(pd.date_range('2015-01-01', '2016-12-31 21:00', freq='3h'))

    dlwp = DLWPFunctional(is_convolutional=True, time_dim=io_time_steps)
    data = xr.open_dataset(predictor_file)
    train_data = data.sel(sample=train_set)
    validation_data = data.sel(sample=validation_set)

    print('Loading data to memory...')
    train_array, input_ind, output_ind, sol = prepare_data_array(train_data, input_sel=io_selection,
                                                                output_sel=io_selection, add_insolation=add_solar)
    generator = ArrayDataGenerator(
    dlwp,
    train_array,
    rank=3,
    input_slice=input_ind,
    output_slice=output_ind,
    input_time_steps=io_time_steps,
    output_time_steps=io_time_steps,
    sequence=integration_steps,
    interval=data_interval,
    # insolation_array=None,
    batch_size=batch_size,
    shuffle=shuffle,
    channels_last=True,
    drop_remainder=True
)
    
    print('Loading validation data to memory...')
    val_array, input_ind, output_ind, sol = prepare_data_array(validation_data, input_sel=io_selection,
                                                            output_sel=io_selection, add_insolation=add_solar)
    val_generator = ArrayDataGenerator(dlwp, val_array, rank=3, input_slice=input_ind, output_slice=output_ind,
                                    input_time_steps=io_time_steps, output_time_steps=io_time_steps,
                                    sequence=integration_steps, interval=data_interval, insolation_array=sol,
                                    batch_size=batch_size, shuffle=False, channels_last=True)
    
    input_names = ['main_input'] + ['solar_%d' % i for i in range(1, integration_steps)]
    tf_train_data = tf_data_generator(generator, batch_size=batch_size, input_names=input_names)
    tf_val_data = tf_data_generator(val_generator, input_names=input_names)

    cs = generator.convolution_shape
    cso = generator.output_convolution_shape
    input_solar = (integration_steps > 1 and add_solar)

    # Define layers. Must be defined outside of model function so we use the same weights at each integration step.
    main_input = Input(shape=cs, name='main_input')
    if input_solar:
        solar_inputs = [Input(shape=None, name='solar_%d' % d) for d in range(1, integration_steps)]
    cube_padding_1 = CubeSpherePadding2D(1, data_format='channels_last')
    pooling_2 = AveragePooling3D((1, 2, 2), data_format='channels_last')
    up_sampling_2 = UpSampling3D((1, 2, 2), data_format='channels_last')
    relu = ReLU(negative_slope=0.1, max_value=10.)
    conv_kwargs = {
        'dilation_rate': 1,
        'padding': 'valid',
        'activation': 'linear',
        'data_format': 'channels_last'
    }
    skip_connections = 'unet' in cnn_model_name.lower()
    conv_2d_1 = CubeSphereConv2D(base_filter_number, 3, **conv_kwargs)
    conv_2d_1_2 = CubeSphereConv2D(base_filter_number, 3, **conv_kwargs)
    conv_2d_1_3 = CubeSphereConv2D(base_filter_number, 3, **conv_kwargs)
    conv_2d_2 = CubeSphereConv2D(base_filter_number * 2, 3, **conv_kwargs)
    conv_2d_2_2 = CubeSphereConv2D(base_filter_number * 2, 3, **conv_kwargs)
    conv_2d_2_3 = CubeSphereConv2D(base_filter_number * 2, 3, **conv_kwargs)
    conv_2d_3 = CubeSphereConv2D(base_filter_number * 4, 3, **conv_kwargs)
    conv_2d_3_2 = CubeSphereConv2D(base_filter_number * 4, 3, **conv_kwargs)
    conv_2d_4 = CubeSphereConv2D(base_filter_number * 4 if skip_connections else base_filter_number * 8, 3, **conv_kwargs)
    conv_2d_4_2 = CubeSphereConv2D(base_filter_number * 8, 3, **conv_kwargs)
    conv_2d_5 = CubeSphereConv2D(base_filter_number * 2 if skip_connections else base_filter_number * 4, 3, **conv_kwargs)
    conv_2d_5_2 = CubeSphereConv2D(base_filter_number * 4, 3, **conv_kwargs)
    conv_2d_5_3 = CubeSphereConv2D(base_filter_number * 4, 3, **conv_kwargs)
    conv_2d_6 = CubeSphereConv2D(base_filter_number if skip_connections else base_filter_number * 2, 3, **conv_kwargs)
    conv_2d_6_2 = CubeSphereConv2D(base_filter_number * 2, 3, **conv_kwargs)
    conv_2d_6_3 = CubeSphereConv2D(base_filter_number * 2, 3, **conv_kwargs)
    conv_2d_7 = CubeSphereConv2D(base_filter_number, 3, **conv_kwargs)
    conv_2d_7_2 = CubeSphereConv2D(base_filter_number, 3, **conv_kwargs)
    conv_2d_7_3 = CubeSphereConv2D(base_filter_number, 3, **conv_kwargs)
    conv_2d_8 = CubeSphereConv2D(cso[-1], 1, name='output', **conv_kwargs)


    

if __name__ == '__main__':
    main()
