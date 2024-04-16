#
# Copyright (c) 2019 Jonathan Weyn <jweyn@uw.edu>
#
# See the file LICENSE for your rights.
#

"""
Extension classes for doing more with models, generators, and so on.
"""

import numpy as np
import xarray as xr
import pandas as pd
from .models import DLWPNeuralNet, DLWPFunctional
from .models_torch import DLWPTorchNN
from .generators import DataGenerator, SeriesDataGenerator, ArrayDataGenerator
from ..util import insolation
import warnings


class TimeSeriesEstimator(object):
    """
    Sophisticated wrapper class for producing time series forecasts from a DLWP model, using a Generator with
    metadata. This class allows predictions with non-matching inputs and outputs, including in the variable, level, and
    time_step dimensions.
    """

    def __init__(self, model, generator):
        """
        Initialize a TimeSeriesEstimator from a model and generator.

        :param model: DLWP model instance
        :param generator: DLWP DataGenerator instance
        """
        if not isinstance(model, (DLWPNeuralNet, DLWPFunctional, DLWPTorchNN)):
            raise TypeError("'model' must be a valid instance of a DLWP model class")
        if not isinstance(generator, (DataGenerator, SeriesDataGenerator, ArrayDataGenerator)):
            raise TypeError("'generator' must be a valid instance of a DLWP generator class")
        if isinstance(model, DLWPFunctional):
            warnings.warn('DLWPFunctional models are only partially supported by TimeSeriesEstimator. The '
                          'inputs/outputs to the model must be the same if the model predicts a sequence.')
        self.model = model
        self.generator = generator
        self._add_insolation = generator._add_insolation if hasattr(generator, '_add_insolation') else False
        self._uses_varlev = 'varlev' in generator.ds.dims
        self._has_targets = 'targets' in generator.ds.variables
        self._is_series = isinstance(generator, (SeriesDataGenerator, ArrayDataGenerator))
        self._output_sel = {}
        self._input_sel = {}
        self._dt = self.generator.ds['sample'][1] - self.generator.ds['sample'][0]
        if hasattr(self.generator, '_interval'):
            self._interval = self.generator._interval
        else:
            self._interval = 1
        if hasattr(self.generator, 'rank'):
            self.rank = self.generator.rank
        else:
            self.rank = 2

        # Generate the selections needed for inputs and outputs
        if self._uses_varlev:
            # Find the outputs we keep
            if self._is_series:  # a SeriesDataGenerator has user-specified I/O
                if not self.generator._output_sel:  # selection was empty
                    self._output_sel = {'varlev': np.array(self.generator.ds.coords['varlev'][:])}
                else:
                    self._output_sel = {k: np.array(v) for k, v in self.generator._output_sel.items()}
            else:
                if self._has_targets:
                    self._output_sel = {'varlev': np.array(self.generator.ds.targets.coords['varlev'][:])}
                else:
                    self._output_sel = {'varlev': np.array(self.generator.ds.coords['varlev'][:])}
            # Find the inputs we need
            if self._is_series:
                if not self.generator._input_sel:  # selection was empty
                    self._input_sel = {'varlev': np.array(self.generator.ds.coords['varlev'][:])}
                else:
                    self._input_sel = {k: np.array(v) for k, v in self.generator._input_sel.items()}
            else:
                self._input_sel = {'varlev': np.array(self.generator.ds.predictors.coords['varlev'][:])}
            # Find outputs that need to replace inputs
            self._outputs_in_inputs = {
                'varlev': np.array([v for v in self._output_sel['varlev'] if v in self._input_sel['varlev']])
            }
        else:  # Uses variable/level coordinates
            # Find the outputs we keep
            if self._is_series:  # a SeriesDataGenerator has user-specified I/O
                if not self.generator._output_sel:  # selection was empty
                    self._output_sel = {'variable': np.array(self.generator.ds.coords['variable'][:]),
                                        'level': np.array(self.generator.ds.coords['level'][:])}
                else:
                    self._output_sel = {k: np.array(v) for k, v in self.generator._output_sel.items()}
                    if 'variable' not in self._output_sel.keys():
                        self._output_sel['variable'] = np.array(self.generator.ds.coords['variable'][:])
                    if 'level' not in self._output_sel.keys():
                        self._output_sel['level'] = np.array(self.generator.ds.coords['level'][:])
            else:
                if self._has_targets:
                    self._output_sel = {'variable': np.array(self.generator.ds.targets.coords['variable'][:]),
                                        'level': np.array(self.generator.ds.targets.coords['level'][:])}
                else:
                    self._output_sel = {'variable': np.array(self.generator.ds.coords['variable'][:]),
                                        'level': np.array(self.generator.ds.coords['level'][:])}
            # Find the inputs we need
            if self._is_series:
                if not self.generator._input_sel:  # selection was empty
                    self._input_sel = {'variable': np.array(self.generator.ds.coords['variable'][:]),
                                       'level': np.array(self.generator.ds.coords['level'][:])}
                else:
                    self._input_sel = {k: np.array(v) for k, v in self.generator._input_sel.items()}
                    if 'variable' not in self._input_sel.keys():
                        self._input_sel['variable'] = np.array(self.generator.ds.coords['variable'][:])
                    if 'level' not in self._input_sel.keys():
                        self._input_sel['level'] = np.array(self.generator.ds.coords['level'][:])
            else:
                self._input_sel = {'variable': np.array(self.generator.ds.predictors.coords['variable'][:]),
                                   'level': np.array(self.generator.ds.predictors.coords['level'][:])}
            # Flatten variable/level
            lev, var = np.meshgrid(self._input_sel['level'], self._input_sel['variable'])
            varlev = np.array(['/'.join([v, str(l)]) for v, l in zip(var.flatten(), lev.flatten())])
            self._input_sel['varlev'] = varlev
            lev, var = np.meshgrid(self._output_sel['level'], self._output_sel['variable'])
            varlev = np.array(['/'.join([v, str(l)]) for v, l in zip(var.flatten(), lev.flatten())])
            self._output_sel['varlev'] = varlev
            # Find outputs that need to replace inputs
            self._outputs_in_inputs = {
                'variable': np.array([v for v in self._output_sel['variable'] if v in self._input_sel['variable']]),
                'level': np.array([v for v in self._output_sel['level'] if v in self._input_sel['level']]),
                'varlev': np.array([v for v in self._output_sel['varlev'] if v in self._input_sel['varlev']])
            }
        if self._add_insolation:
            self._input_sel['varlev'] = np.concatenate([self._input_sel['varlev'], np.array(['SOL'])])

        # Time step dimension
        self._input_time_steps = (generator._input_time_steps if isinstance(generator, SeriesDataGenerator)
                                  else model.time_dim)
        self._output_time_steps = (generator._output_time_steps if isinstance(generator, SeriesDataGenerator)
                                   else model.time_dim)

        # Channels last option
        self.channels_last = hasattr(self.generator, 'channels_last') and self.generator.channels_last
        self._forward_transpose = (0, self.rank + 1) + tuple(range(1, 1 + self.rank)) + (-1,)
        self._backward_transpose = (0,) + tuple(range(2, 2 + self.rank)) + (1, -1,)

        # Constants in the generator
        if hasattr(self.generator, 'constants') and self.generator.constants is not None:
            if self.channels_last:
                self.constants = self.generator.constants.transpose(tuple(range(1, 1 + self.rank)) + (0,))
            else:
                self.constants = self.generator.constants
        else:
            self.constants = None

    @property
    def shape(self):
        return (self.generator._n_sample,) + self.generator.shape

    @property
    def convolution_shape(self):
        return (self.generator._n_sample,) + self.generator.convolution_shape

    def predict(self, steps, samples=(), impute=False, keep_time_dim=False, prefer_first_times=True,
                f_hour_timedelta_type=False, **kwargs):
        """
        Step forward the time series prediction from the model 'steps' times, feeding predictions back in as
        inputs. Predicts for all the data provided in the generator. If there are inputs which are not produced by
        the model outputs, we include the available inputs from the generator data and either reduce the number of
        predicted samples accordingly (remove those whose inputs cannot be satisfied) or run the model using the mean
        values of the inputs which cannot be satisfied. If there are fewer output time steps than input time steps,
        then we build a time series forecast intelligently using part of the predictors and part of the prediction at
        every step. Note only the SeriesDataGenerator supports variable inputs/outputs.

        :param steps: int: number of times to step forward
        :param samples: list of int: which samples in the generator to predict for. For all samples, pass an empty
            list or tuple. May cause unexpected behavior when the input and output data do not match.
        :param impute: bool: if True, use the mean state for missing inputs in the forward integration
        :param keep_time_dim: bool: if True, keep the time_step dimension instead of integrating it with forecsat_hour
            to produce a continuous time series
        :param prefer_first_times: bool: in the case where the prediction contains more time_steps than the input,
            use the first available predicted times to initialize the next step, otherwise use the last times. If the
            output time_steps is less than the input time_steps, we always use all of the output times.
        :param f_hour_timedelta_type: bool: if True, converts f_hour dimension into a timedelta type. May not always be
            compatible with netCDF applications.
        :param kwargs: passed to Keras.predict()
        :return: ndarray: predicted states with forecast_step as the first dimension
        """
        if int(steps) < 1:
            raise ValueError('must use positive integer for steps')

        # Effective forward time steps for each step
        if self._output_time_steps <= self._input_time_steps:
            keep_inputs = True
            es = self._output_time_steps
            in_times = np.arange(self._input_time_steps) - (self._input_time_steps - self._output_time_steps)
        else:
            keep_inputs = False
            if prefer_first_times:
                es = self._input_time_steps
                in_times = np.arange(self._input_time_steps)
            else:
                es = self._output_time_steps
                in_times = np.arange(self._input_time_steps) + (self._output_time_steps - self._input_time_steps)
        effective_steps = int(np.ceil(steps / es))

        # Load data from the generator, without any scaling as this will be done by the model's predict method
        predictors, t = self.generator.generate(samples, scale_and_impute=False)
        p = predictors[0] if isinstance(predictors, (list, tuple)) else predictors
        p_shape = tuple(p.shape)

        # Add metadata. The sample dimension denotes the *start* time of the sample, for insolation purposes. This is
        # then corrected when providing the final output time series.
        sample_coord = self.generator.ds.sample[:self.generator._n_sample] if len(samples) == 0 else \
            self.generator.ds.sample[samples]
        if not self._is_series:
            sample_coord = sample_coord - self._dt * (self._input_time_steps - 1)
        if self.channels_last:
            # Split time step/varlev at the end and transpose
            if not self.generator._keep_time_axis:
                p = p.reshape((p_shape[0],) + self.generator.convolution_shape[-self.rank-1:-1] +
                              (self._input_time_steps, -1)).transpose(self._forward_transpose)
            p_da = xr.DataArray(
                p,
                coords=([sample_coord, in_times] +
                        [np.arange(d) for d in self.generator.convolution_shape[-self.rank-1:-1]] +
                        [self._input_sel['varlev']]),
                dims=['sample', 'time_step'] + ['x%d' % d for d in range(self.rank)] + ['varlev']
            )
        else:
            p = p.reshape((p_shape[0], self._input_time_steps, -1,) + self.generator.convolution_shape[-self.rank:])
            p_da = xr.DataArray(
                p,
                coords=([sample_coord, in_times, self._input_sel['varlev']] +
                        [np.arange(d) for d in self.generator.convolution_shape[-self.rank:]]),
                dims=['sample', 'time_step', 'varlev'] + ['x%d' % d for d in range(self.rank)]
            )
        if self.rank == 2:
            p_da = p_da.rename({'x0': 'lat', 'x1': 'lon'}).assign_coords(lat=self.generator.ds.lat,
                                                                         lon=self.generator.ds.lon)

        # Calculate mean for imputing
        if impute:
            p_mean = p.mean(axis=0)

        # Target shape
        t_shape = t[0].shape if isinstance(t, (list, tuple)) else t.shape
        t = None

        # A DLWPFunctional model which does not have the same inputs/outputs must have some programmatic way of fixing
        # this issue built-in (if it is trained to minimize several iterations of the model). Thus we fall back to the
        # regular predict_timeseries method. However, such a model that does not predict a sequence is perfectly valid
        # for use here.
        if isinstance(self.model, DLWPFunctional) and self.model._n_steps > 1:
            if not self.generator._add_insolation:
                # For the DLWPFunctional model, just use its predict_timeseries API, which handles effective steps
                # TODO: correctly handle channels_last
                result = self.model.predict_timeseries(predictors, steps, keep_time_dim=True,
                                                       **kwargs).reshape((-1,) + t_shape)[:effective_steps, ...]
                result = result.transpose((0, 1) + tuple(range(2, len(result.shape)))).copy()
            else:
                # If insolation is requested, intelligently add it in the same way the generator does
                sequence_steps = int(np.ceil(steps / self.model._n_steps / self.model.time_dim))

                # Giant forecast array
                result = np.full((t_shape[0], sequence_steps, self.model._n_steps) + t_shape[1:],
                                 np.nan, dtype=np.float32)

                # Iterate
                new_t = p_da.sample[:]
                for s in range(sequence_steps):
                    if 'verbose' in kwargs and kwargs['verbose'] > 0:
                        print('Time step %d/%d' % (s + 1, sequence_steps))
                    result[:, s] = np.stack(self.model.predict(predictors, **kwargs), axis=1)

                    # Assign new insolation to list of inputs
                    new_t = new_t + self._output_time_steps * self.model._n_steps * self._dt
                    new_insolation = [np.concatenate(
                        [np.expand_dims(insolation(new_t + (n + m * self._input_time_steps) * self._dt,
                                                   self.generator.ds.lat.values,
                                                   self.generator.ds.lon.values)[:, None],
                                        axis=-1 if self.channels_last else 1)
                         for n in range(self._input_time_steps)],
                        axis=1) for m in range(self.model._n_steps)]

                    if self.channels_last:
                        if self.generator._keep_time_axis:
                            r = result[:, s, -1]
                            predictors = [np.concatenate([r, new_insolation[0]], axis=-1).reshape(
                                (-1,) + self.convolution_shape[1:])] + new_insolation[1:]
                        else:
                            r = result[:, s, -1].reshape(t_shape[:-1] + (self._output_time_steps, -1)).transpose(
                                self._forward_transpose
                            )
                            predictors = [np.concatenate([r, new_insolation[0]], axis=-1).transpose(
                                self._backward_transpose).reshape((-1,) + self.convolution_shape[1:])] \
                                + new_insolation[1:]
                    else:
                        predictors = [
                            np.concatenate([result[:, s, -1].reshape((-1,) + self.shape[1:]), new_insolation[0]],
                                           axis=2).reshape((-1,) + self.convolution_shape[1:])
                        ] + new_insolation[1:]

                    # Add constants
                    if self.constants is not None:
                        predictors.append(np.repeat(np.expand_dims(self.constants, axis=0), len(p_da.sample), axis=0))

                n_dim_1 = result.size // int(np.prod(t_shape))
                result.shape = (t_shape[0], n_dim_1,) + t_shape[1:]
                result = result[:, :effective_steps, ...]
        else:
            # Giant forecast array
            result = np.full((t_shape[0], effective_steps,) + t_shape[1:], np.nan, dtype=np.float32)

            # Iterate prediction forward for a regular DLWP Sequential NN
            for s in range(effective_steps):
                if 'verbose' in kwargs and kwargs['verbose'] > 0:
                    print('Time step %d/%d' % (s + 1, effective_steps))
                if self.channels_last and not self.generator._keep_time_axis:
                    result[:, s] = self.model.predict(p_da.values.transpose(self._backward_transpose).reshape(p_shape),
                                                      **kwargs)
                else:
                    result[:, s] = self.model.predict(p_da.values.reshape(p_shape), **kwargs)

                # Add metadata to the prediction
                if self.channels_last:
                    if not self.generator._keep_time_axis:
                        r = result[:, s].reshape((p_shape[0],) + self.generator.convolution_shape[-self.rank-1:-1] +
                                                 (self._output_time_steps, -1)).transpose(self._forward_transpose)
                    else:
                        r = result[:, s]
                    r_da = xr.DataArray(
                        r,
                        coords=([p_da.sample + (es + self._interval - 1) * self._dt,
                                 np.arange(self._output_time_steps)] +
                                [np.arange(d) for d in self.generator.convolution_shape[-self.rank-1:-1]] +
                                [self._output_sel['varlev']]),
                        dims=['sample', 'time_step'] + ['x%d' % d for d in range(self.rank)] + ['varlev']
                    )
                else:
                    r_da = xr.DataArray(
                        result[:, s].reshape((p_shape[0], self._output_time_steps, -1,) +
                                             self.generator.convolution_shape[-self.rank:]),
                        coords=([p_da.sample + (es + self._interval - 1) * self._dt,
                                 np.arange(self._output_time_steps), self._output_sel['varlev']] +
                                [np.arange(d) for d in self.generator.convolution_shape[-self.rank:]]),
                        dims=['sample', 'time_step', 'varlev'] + ['x%d' % d for d in range(self.rank)]
                    )
                if self.rank == 2:
                    r_da = r_da.rename({'x0': 'lat', 'x1': 'lon'}).assign_coords(lat=self.generator.ds.lat,
                                                                                 lon=self.generator.ds.lon)

                # Re-index the predictors to the new forward time step
                p_da = p_da.reindex(sample=r_da.sample, method=None)

                # Impute values extending beyond data availability
                if impute:
                    # Calculate mean values for the added time steps after re-indexing
                    p_da[-es:] = np.concatenate([p_mean[np.newaxis, ...]] * es)

                # Take care of the known insolation for added time steps
                if self._add_insolation:
                    p_da.loc[{'varlev': 'SOL'}][-es:] = \
                        np.concatenate([insolation(p_da.sample[-es:] + n * self._dt, self.generator.ds.lat.values,
                                                   self.generator.ds.lon.values)[:, np.newaxis]
                                        for n in range(self._input_time_steps)], axis=1)

                # Replace the predictors that exist in the result with the result. Any that do not exist are
                # automatically inherited from the known predictor data (or imputed data).
                if keep_inputs:
                    loc_dict = dict(varlev=self._outputs_in_inputs['varlev'], time_step=p_da.time_step[-es:])
                    p_da.loc[loc_dict] = r_da.loc[{'varlev': self._outputs_in_inputs['varlev']}]
                else:
                    if prefer_first_times:
                        p_da.loc[{'varlev': self._outputs_in_inputs['varlev']}] = \
                            r_da.loc[{'varlev': self._outputs_in_inputs['varlev']}][:, :self._input_time_steps]
                    else:
                        p_da.loc[{'varlev': self._outputs_in_inputs['varlev']}] = \
                            r_da.loc[{'varlev': self._outputs_in_inputs['varlev']}][:, -self._input_time_steps:]

        # Return a DataArray. Keep the actual model initialization, that is, the last available time in the inputs,
        # as the time
        rv = result.view()
        if self.channels_last:
            if not self.generator._keep_time_axis:
                rv.shape = (p_shape[0], effective_steps,) + \
                           self.generator.output_convolution_shape[-self.rank-1:-1] + (self._output_time_steps, -1,)
        else:
            rv.shape = (p_shape[0], effective_steps, self._output_time_steps, -1,) + \
                self.generator.output_convolution_shape[-self.rank:]
        if f_hour_timedelta_type:
            dt = self._dt.values
        else:
            dt = np.array(self._dt.values.astype('timedelta64[h]').astype('float'))
        if keep_time_dim:
            if self.channels_last:
                result_da = xr.DataArray(
                    rv.transpose((1, 0) + tuple(range(2, len(rv.shape)))) if self.generator._keep_time_axis else
                    rv.transpose((1, 0, -2) + tuple(range(2, 2 + self.rank)) + (-1,)),
                    coords=[
                               np.arange(dt, (effective_steps * (es + self._interval - 1) + 1) * dt,
                                         (es + self._interval - 1) * dt),
                               sample_coord + (self._input_time_steps - 1) * self._dt,
                               range(self._output_time_steps),
                           ]
                    + [np.arange(d) for d in self.generator.output_convolution_shape[-self.rank-1:-1]]
                    + [self._output_sel['varlev']],
                    dims=['f_hour', 'time', 'time_step'] + ['x%d' % d for d in range(self.rank)] + ['varlev'],
                    name='forecast'
                )
            else:
                result_da = xr.DataArray(
                    rv.transpose((1, 0) + tuple(range(2, len(rv.shape)))),
                    coords=[
                        np.arange(dt, (effective_steps * (es + self._interval - 1) + 1) * dt,
                                  (es + self._interval - 1) * dt),
                        sample_coord + (self._input_time_steps - 1) * self._dt,
                        range(self._output_time_steps),
                        self._output_sel['varlev'],
                    ] + [np.arange(d) for d in self.generator.output_convolution_shape[-self.rank:]],
                    dims=['f_hour', 'time', 'time_step', 'varlev'] + ['x%d' % d for d in range(self.rank)],
                    name='forecast'
                )
            if self.rank == 2:
                result_da = result_da.rename({'x0': 'lat', 'x1': 'lon'}).assign_coords(lat=self.generator.ds.lat,
                                                                                       lon=self.generator.ds.lon)
        else:
            # To create a correct time series, we must retain only the effective steps
            if not keep_inputs:
                if prefer_first_times:
                    rv = rv[:, :, :es]
            if self.channels_last:
                if self.generator._keep_time_axis:
                    rv.shape = (t_shape[0], rv.shape[1] * rv.shape[2]) + rv.shape[3:]
                result_da = xr.DataArray(
                    rv.transpose((1, 0) + tuple(range(2, len(rv.shape)))) if self.generator._keep_time_axis else
                    rv.transpose((1, -2, 0) + tuple(range(2, 2 + self.rank)) + (-1,)).reshape(
                        (rv.shape[1] * rv.shape[-2], rv.shape[0]) +
                        self.generator.output_convolution_shape[-self.rank-1:-1] + (-1,)
                    ),
                    coords=[
                               np.array([(np.arange(0, es) + self._interval + e * (es - 1 + self._interval)) * dt
                                         for e in range(effective_steps)]).flatten(),
                               sample_coord + (self._input_time_steps - 1) * self._dt,
                           ]
                    + [np.arange(d) for d in self.generator.output_convolution_shape[-self.rank-1:-1]]
                    + [self._output_sel['varlev']],
                    dims=['f_hour', 'time'] + ['x%d' % d for d in range(self.rank)] + ['varlev'],
                    name='forecast'
                )
            else:
                rv.shape = (t_shape[0], rv.shape[1] * rv.shape[2]) + rv.shape[3:]
                result_da = xr.DataArray(
                    rv.transpose((1, 0) + tuple(range(2, len(rv.shape)))),
                    coords=[
                        np.array([(np.arange(0, es) + self._interval + e * (es - 1 + self._interval)) * dt
                                  for e in range(effective_steps)]).flatten(),
                        sample_coord + (self._input_time_steps - 1) * self._dt,
                        self._output_sel['varlev'],
                    ] + [np.arange(d) for d in self.generator.output_convolution_shape[-self.rank:]],
                    dims=['f_hour', 'time', 'varlev'] + ['x%d' % d for d in range(self.rank)],
                    name='forecast'
                )
            if self.rank == 2:
                result_da = result_da.rename({'x0': 'lat', 'x1': 'lon'}).assign_coords(lat=self.generator.ds.lat,
                                                                                       lon=self.generator.ds.lon)
            result_da = result_da.isel(f_hour=slice(0, steps))

        # Expand back out to variable/level pairs
        if self._uses_varlev:
            return result_da
        else:
            var, lev = self._output_sel['variable'], self._output_sel['level']
            vl = pd.MultiIndex.from_product((var, lev), names=('variable', 'level'))
            result_da = result_da.assign_coords(varlev=vl).unstack('varlev')
            spatial_dims = [d for d in result_da.dims if d not in ['f_hour', 'time', 'variable', 'level']]
            if self.channels_last:
                transpose_dims = ('f_hour', 'time') + tuple(spatial_dims) + ('variable', 'level')
                result_da = result_da.transpose('f_hour', 'time', 'lat', 'lon', 'variable', 'level')
            else:
                transpose_dims = ('f_hour', 'time') + ('variable', 'level') + tuple(spatial_dims)
            result_da = result_da.transpose(*transpose_dims)
            return result_da


class SeriesDataGeneratorWithInference(SeriesDataGenerator):
    """
    An extension of the SeriesDataGenerator that couples a second model for inference of part of the sequence target
    data. For example, with a model that predicts a sequence, a fixed model with fixed weights can be used to predict
    the first step of a sequence while a new model is trained on the existing model first step and then real data for
    the following steps.
    """

    def __init__(self, inference_model, inference_steps, *args, **kwargs):
        """
        Initialize an SeriesDataGenerator with an inference model. The arguments and kwargs must be those passed to the
        ArrayDataGenerator. The parameter inference_steps governs which parts of the target sequence are replaced with
        the inference model prediction. Numbers in inference_steps should be integers starting with 0 that correspond
        to which steps in the target data should be replaced my the inference model prediction.

        :param inference_model: DLWP model
        :param inference_steps: iterable: steps in sequence to replace with inference
        :param args: passed to SeriesDataGenerator()
        :param kwargs: passed to SeriesDataGenerator()
        """
        if not isinstance(inference_model, (DLWPNeuralNet, DLWPFunctional, DLWPTorchNN)):
            raise TypeError("inference model should be a DLWP model")

        self.inference_model = inference_model
        super(SeriesDataGeneratorWithInference, self).__init__(*args, **kwargs)

        if self._sequence is None:
            raise ValueError("'SeriesDataGeneratorWithInference' is only usable if the generator produces a "
                             "sequence of targets")
        if np.any(np.array(inference_steps)) < 0 or np.any(np.array(inference_steps)) >= self._sequence:
            raise ValueError("got steps parameter (%s) outside of range 0 to %s" % (inference_steps, self._sequence))
        self.inference_steps = inference_steps

    def generate(self, samples, scale_and_impute=True):
        # Generate data normally
        p, t = super(SeriesDataGeneratorWithInference, self).generate(samples, scale_and_impute)

        # Make a prediction with the inference model
        predicted = self.inference_model.predict(p)

        # Insert inference prediction
        for s in self.steps:
            t[s] = predicted[s][:]

        # Return modified sample
        return p, t


class ArrayDataGeneratorWithInference(ArrayDataGenerator):
    """
    An extension of the ArrayDataGenerator that couples a second model for inference of part of the sequence target
    data. For example, with a model that predicts a sequence, a fixed model with fixed weights can be used to predict
    the first step of a sequence while a new model is trained on the existing model first step and then real data for
    the following steps.
    """

    def __init__(self, inference_model, inference_steps, *args, **kwargs):
        """
        Initialize an ArrayDataGenerator with an inference model. The arguments and kwargs must be those passed to the
        ArrayDataGenerator. The parameter inference_steps governs which parts of the target sequence are replaced with
        the inference model prediction. Numbers in inference_steps should be integers starting with 0 that correspond
        to which steps in the target data should be replaced my the inference model prediction.

        :param inference_model: DLWP model
        :param inference_steps: iterable: steps in sequence to replace with inference
        :param args: passed to ArrayDataGenerator()
        :param kwargs: passed to ArrayDataGenerator()
        """
        if not isinstance(inference_model, (DLWPNeuralNet, DLWPFunctional, DLWPTorchNN)):
            raise TypeError("inference model should be a DLWP model")

        self.inference_model = inference_model
        super(ArrayDataGeneratorWithInference, self).__init__(*args, **kwargs)

        if self._sequence is None:
            raise ValueError("'ArrayDataGeneratorWithInference' is only usable if the generator produces a "
                             "sequence of targets")
        if np.any(np.array(inference_steps)) < 0 or np.any(np.array(inference_steps)) >= self._sequence:
            raise ValueError("got steps parameter (%s) outside of range 0 to %s" % (inference_steps, self._sequence))
        self.inference_steps = inference_steps

    def generate(self, samples):
        # Generate data normally
        p, t = super(ArrayDataGeneratorWithInference, self).generate(samples)

        # Make a prediction with the inference model
        predicted = self.inference_model.predict(p)

        # Insert inference prediction
        for s in self.inference_steps:
            t[s] = predicted[s][:]

        # Return modified sample
        return p, t
