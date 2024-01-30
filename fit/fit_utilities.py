import sys

import uproot
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import glob

from lmfit.models import ExponentialGaussianModel, ConstantModel, GaussianModel
from lmfit import minimize, Parameters, report_fit


#Move peak to a specific time
def move_x_to_peak(x, y, peak=100, restrict=None):
    #Assume peak is the minimum
    min_index = np.argmin(y)

    x_diff = peak - x[min_index]

    return [i+x_diff for i in x]

#Average waveforms
def average(x_array,y_array, common_time = 100, bunch=None, spill=None):



    min_index_array = []
    right_array = []


    #Find which waveforms are longer before peak (min_index) and which are longer after peak (right_array)
    for waveform_x, waveform_y in zip(x_array,y_array):
        min_index_array.append(np.argmin(waveform_y))
        right_array.append(len(waveform_y) - np.argmin(waveform_y))

    largest_min_index = np.max(min_index_array)
    smallest_right_index = np.max(right_array)
    if spill==5 and bunch==4:
        print(f"largest min index: {largest_min_index}, smallest right index: {smallest_right_index}")

    new_x_array = []
    new_y_array = []
    new_mask_array = []
    #Make all arrays same length
    i=0
    for waveform_x, waveform_y, min_idx, right_idx in zip(x_array, y_array, min_index_array, right_array):
        temp_min_index = np.argmin(waveform_y)
        temp_right_index = len(waveform_y) - np.argmin(waveform_y)
        temp_x_array = waveform_x
        temp_y_array = waveform_y
        if spill==5 and bunch==4:
            print(f"temp_right_index: {temp_right_index}, temp_min_index: {temp_min_index}")
            print(list(waveform_x))
        if smallest_right_index > temp_right_index:
            temp_x_array = np.append(temp_x_array, np.zeros(smallest_right_index-temp_right_index))
            temp_y_array = np.append(temp_y_array, np.zeros(smallest_right_index-temp_right_index))
        if largest_min_index > temp_min_index:
            temp_x_array = np.append(np.zeros(largest_min_index-temp_min_index),temp_x_array)
            temp_y_array = np.append(np.zeros(largest_min_index-temp_min_index),temp_y_array)


        new_x_array.append(temp_x_array)
        new_y_array.append(temp_y_array)
        new_mask_array.append((new_y_array[i]!=0).astype(int))
        if spill==5 and bunch==4:
            print(list((new_y_array[i]!=0).astype(int)))
        i=i+1

    avg_y = np.zeros(len(new_y_array[0]))
    avg_x = np.zeros(len(new_x_array[0]))
    sum_mask = np.zeros(len(new_mask_array[0]))

    for new_x, new_y, new_mask in zip(new_x_array, new_y_array, new_mask_array):
        avg_y = np.add(avg_y, new_y)
        if spill==5 and bunch==4:
            print(list(new_x))
        avg_x = np.add(avg_x, new_x)
        sum_mask = np.add(sum_mask,new_mask)

    avg_x = np.divide(avg_x,sum_mask)
    avg_y = np.divide(avg_y,sum_mask)


    return avg_x, avg_y

    #print(min_index_array)
    #print(right_array)

def make_waveform(data, timescale, xlabel, ylabel, name, x_range=None, y_range=None, fit_val=None, fit_names = None, fit_chi=None, savefig=False, showfig=False):
    fig, ax = plt.subplots()
    #Time scale is either int that is time between (assume we start at 0) or list
    if isinstance(timescale, int):
        time = list(range(0,len(data)*timescale, timescale))
    else:
        time = timescale
    print(time.shape)
    print(data.shape)
    ax.plot(time, data, 'o', label='data')

    if x_range is not None:
        ax.set_xlim(x_range[0], x_range[1])

    if y_range is not None:
        ax.set_ylim(y_range[0], y_range[1])

    if fit_val is not None:
        if fit_chi is not None:
            colors = ['Orange', 'Red', 'Green', 'Black', 'Purple']
            for i, fit in enumerate(fit_val):
                ax.plot(time, fit, '-', label=f'{fit_names[i]} ({fit_chi[i]:.3f})', color = colors[i])
        else:
            for i, fit in enumerate(fit_val):
                ax.plot(time, fit, '-', label=f'{fit_names[i]}: ({res:.3f})', color=colors[i])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.legend()
    if showfig:
        plt.show()
    if savefig:
        plt.savefig(name)
    plt.close()



def make_ct_like_otr(x,y):
    return 2160 - (500*y)

def fit_waveform(function, x, y, initial_parameters, maxfev=1000, function_name="Generic", parameter_names=None, verbose=True, bounds = None, add_uncertainty = True):

    if bounds is None:
        bounds_lower = []
        bounds_upper = []
        for i in range(0,len(initial_parameters)):
            bounds_lower.append(-np.inf)
            bounds_upper.append(np.inf)
        bounds = [bounds_lower, bounds_upper]

    if add_uncertainty:
        sigma = 20*np.ones(len(x))
    else:
        sigma = None

    #parameters, covariance, info, _, __ = curve_fit(function, x, y, p0=initial_parameters, bounds=bounds, maxfev=maxfev, full_output=True, sigma = sigma)
    if False and verbose and parameter_names is not None and len(parameter_names) == len(parameters):
        print(f'{function_name} ', end='')
        for i, parameter_name in enumerate(parameter_names):
            print(f', {parameter_name}: {parameters[i]:.3f}', end='')
        print('\n')

    #Clunky but OK
    bias=2160
    if function == Gauss:
        #fit_y = function(x, parameters[0], parameters[1], parameters[2], parameters[3])
        #bias = parameters[3]
        model = GaussianModel() + ConstantModel()
        parameters = model.make_params(amplitude={'value':-10000, 'min':-50000, 'max':50000},
                                        center={'value':100, 'min':90, 'max':110},
                                        c={'value':2160, 'min':2000, 'max':2200},
                                        sigma={'value':12, 'min':8, 'max':20})
        result = model.fit(y,parameters, x=x, max_nfev=100000)
        fit_y = result.best_fit
    if function == DoubleGauss:
        #fit_y = function(x, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6])
        #bias = parameters[6]
        model = GaussianModel() + GaussianModel(prefix='dg_') + ConstantModel()
        parameters = model.make_params(amplitude={'value':-10000, 'min':-50000, 'max':50000},
                                        center={'value':95, 'min':90, 'max':110},
                                        c={'value':2160, 'min':2000, 'max':2200},
                                        sigma={'value':12, 'min':8, 'max':20},
                                        dg_amplitude={'value':-10000, 'min':-50000, 'max':50000},
                                        dg_center={'value':100, 'min':90, 'max':110},
                                        dg_sigma={'value':12, 'min':8, 'max':20})
        result = model.fit(y,parameters, x=x, max_nfev=100000)
        fit_y = result.best_fit
    if function == TripleGauss:
        fit_y = function(x, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9])
        bias = parameters[9]
    if function == Convol:
        #fit_y = function(x, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6])
        #bias = parameters[6]
        model = ExponentialGaussianModel() + ConstantModel()
        parameters = model.make_params(amplitude={'value':-10000, 'min':-50000, 'max':50000},
                                        center={'value':95, 'min':90, 'max':110},
                                        c={'value':2160, 'min':2000, 'max':2200},
                                        sigma={'value':12, 'min':8, 'max':20},
                                        gamma={'value':0.1, 'min':0.01, 'max':1.0})
        result = model.fit(y,parameters, x=x, max_nfev=100000)
        fit_y = result.best_fit
    if function == DoublegaussConvol:
        #fit_y = function(x, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8])
        #bias = parameters[8]
        model = ExponentialGaussianModel() + ConstantModel() + GaussianModel(prefix='dg_')
        parameters = model.make_params(amplitude={'value':-10000, 'min':-50000, 'max':50000},
                                        center={'value':95, 'min':90, 'max':110},
                                        c={'value':2160, 'min':2000, 'max':2200},
                                        sigma={'value':12, 'min':8, 'max':20},
                                        dg_amplitude={'value':-10000, 'min':-50000, 'max':50000},
                                        dg_center={'value':95, 'min':90, 'max':110},
                                        dg_sigma={'value':12, 'min':8, 'max':20},
                                        gamma={'value':0.05, 'min':0.001, 'max':1.2})
        print(parameters)
        result = model.fit(y,parameters, x=x, max_nfev=100000)
        print(result.fit_report())
        fit_y = result.best_fit
    if function == DoublegaussConvolBackground:
        fit_y = function(x, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6])
        bias = parameters[6]



    chi_square = np.sum(np.divide(np.square(np.subtract(y,fit_y)),np.abs(np.add(np.subtract(bias,y),.1))))
    dof = len(x) - len(parameters)
    chi_per_dof = chi_square/dof
    res=0

    return fit_y, chi_per_dof, res, parameters



