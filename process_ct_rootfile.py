#Converts root file from CT oscilloscope data

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

def convolve_custom(arr, kernel):
    """Simple convolution of two arrays."""
    npts = min(arr.size, kernel.size)
    pad = np.ones(npts)
    tmp = np.concatenate((pad*arr[0], arr, pad*arr[-1]))
    out = np.convolve(tmp, kernel, mode='valid')
    noff = int((len(out) - npts) / 2)
    return out[noff:noff+npts]

# Define the Gaussian function
def Gauss(x, a, x0, sigma,bias):
    return bias+(a/sigma)*np.exp(-(x-x0)**2/(2*sigma**2))

def DoubleGauss(x, a, x0, sigma0, b,x1, sigma1,bias):
    return bias+(a/sigma0)*np.exp(-(x-x0)**2/(2*sigma0**2))+(b/sigma1)*np.exp(-(x-x1)**2/(2*sigma1**2))

def TripleGauss(x, a, x0, sigma0, b,x1, sigma1, c, x2, sigma2, bias):
    return bias+a*np.exp(-(x-x0)**2/(2*sigma0**2))+b*np.exp(-(x-x1)**2/(2*sigma1**2))+c*np.exp(-(x-x2)**2/(2*sigma2**2))

def Convol(x, a, x0, sigma,bias, tau, x1, b):
    return bias + convolve_custom((a/sigma)*np.exp(-(x-x0)**2/(2*sigma**2)), (b/tau)*np.exp(-(x-x0)/tau))

#def DoublegaussConvol(x, a, x0, sigma0, b, x1, sigma1, tau, x2, c, bias):
#    return bias + (c/(tau*sigma0))*convolve_custom(np.exp(-(x-x0)**2/(2*sigma0**2)), np.exp(-(x-x2)/tau)) + (b/sigma1)*np.exp(-(x-x1)**2/(2*sigma1**2))

def DoublegaussConvol(x, a, x0, sigma0, b, sigma1, x1, tau, x2, bias):
    return bias + (a/(tau*sigma0))*convolve_custom(np.exp(-(x-x0)**2/(2*sigma0**2)), np.exp(-(x-x2)/tau)) + (b/sigma1)*np.exp(-(x-x1)**2/(2*sigma1**2))


def DoublegaussConvolBackground(x, a, x0, sigma0, sigma1, tau0, tau1, bias):
    return bias + (a/(sigma0*sigma1*tau0*tau1))*convolve_custom(convolve_custom(np.exp(-(x-x0)**2/(2*sigma0**2)), np.exp(-(x-x0)/tau0)) + np.exp(-(x-x0)**2/(2*sigma1**2)), np.exp(-(x-x0)/tau1))

def move_x_to_peak(x, y, peak=100):
    #Assume peak is the minimum
    min_index = np.argmin(y)

    x_diff = peak - x[min_index]

    return [i+x_diff for i in x]

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

rootfiles = glob.glob('/home/t2k_otr/data_ct/tek*.root')
adc_rootfile = glob.glob('/home/t2k_otr/data_ct/run0910253_gen.root')

sigma = []

data_adc = uproot.open(adc_rootfile[0]+':anabeam')
print(np.array(data_adc['ctadc'].array()).shape)

ct02 = (data_adc['ctadc'].array(library="np"))[14:23,4,250:350]
print(len(range(0,4*len(ct02[1]), 4)))
print(ct02[0,:].shape)
plt.scatter(range(0,4*len(ct02[1]), 4), ct02[8,:], label='data')
plt.legend()
plt.savefig(f"plots/ct/0_ctadc.png")
plt.close()

B = []
C = []
tau = []
sigma0 = []
sigma1 = []

for i, rootfile in enumerate(rootfiles):
    data = uproot.open(rootfile+':g_ch1')
    #To ns
    x = np.array(data.values()[0])*(10**9)
    y = np.array(data.values()[1])
    y = make_ct_like_otr(x,y)
    x = np.array(move_x_to_peak(x,y))
    time_start = [[-1200, -800, -100, 300, 900, 1500, 2200, 2700],
                    [-3800,-3200,-2600,-2000,-1400,-800,-200,400],
                    [-800, -100, 300, 900, 1500, 2200, 2700,3200],
                    [-1800,-1200, -800, -100, 300, 900, 1500, 2200],
                    [-100, 300, 900, 1500, 2200, 2700, 3300, 3900],
                    [-3600,-3000,-2400,-1800,-1200, -800, -100, 300],
                    [-1200, -800, -100, 300, 900, 1500, 2200, 2700],
                    [-100, 300, 900, 1500, 2200, 2700, 3300, 3900],
                    [-3000,-2400,-1800,-1200, -800, -100, 300, 900],
                    [-3000,-2400,-1800,-1200, -800, -100, 300, 900]]
    time_end = [[-800, -200, 300, 900, 1500, 2200, 2700, 3300],
                    [-3200,-2600,-2000,-1400,-800,-200,400,1000],
                    [-200, 300, 900, 1500, 2200, 2700, 3200, 3800],
                    [-1200,-800, -200, 300, 900, 1500, 2200, 2700],
                    [300, 900, 1500, 2200, 2700, 3300, 3900, 4500],
                    [-3000,-2400,-1800,-1200,-800, -200, 300, 900],
                    [-800, -200, 300, 900, 1500, 2200, 2700, 3300],
                    [300, 900, 1500, 2200, 2700, 3300, 3900,4500],
                    [-2400,-1800,-1200,-800, -200, 300, 900, 1500],
                    [-2400,-1800,-1200,-800, -200, 300, 900, 1500]]


    for j in range(0,8):

        print(f"Bunch {j}")

        temp_y = y[(x > time_start[i][j]) & (x < time_end[i][j])]
        temp_x = x[(x > time_start[i][j]) & (x < time_end[i][j])]

        temp_x = np.array(move_x_to_peak(temp_x,temp_y, peak=100))

        avg_frq = 1

        
        if j > -1:

            temp_prev_y = y[(x > time_start[i][j-1]) & (x < time_end[i][j-1])]
            temp_prev_x = x[(x > time_start[i][j-1]) & (x < time_end[i][j-1])]

            temp_prev_x = np.array(move_x_to_peak(temp_prev_x,temp_prev_y, peak=100))

            avg_x, avg_y = average([temp_x, temp_prev_x],[temp_y, temp_prev_y], common_time = 100, spill=i, bunch=j)

            avg_y = avg_y[(avg_x > 0) & (avg_x < 250)]
            avg_x = avg_x[(avg_x > 0) & (avg_x < 250)]

            temp_x = avg_x
            temp_y = avg_y
            plt.scatter(temp_x, temp_y, label='data')
            plt.legend()
            plt.savefig(f"plots/ct/average_spill{i}_bunch{j}_ct.png")
            plt.close()
        else: 
            continue


        #plt.scatter(temp_x, temp_y, label='data')
        #plt.legend()
        #plt.savefig(f"plots/ct/spill{i}_bunch{j}_ct.png")
        #plt.close()

        #continue

        #Gaussian
        #Amplitude, x0, sigma, bias
        guess = [-500, 100, 12, 2160]
        maxfev=10000
        fit_y, chi_square_gauss, res, parameters = fit_waveform(Gauss, temp_x, temp_y, guess, maxfev, function_name = "Gaussian", parameter_names = ['A', 'x0', 'sigma', 'bias'])

        #Double Gaussian
        #A, x0, sigma0, B,x1, sigma1,bias):
        guess = [0, 80, 1, -5000, 100, 15, 2160]
        bounds_lower = [-np.inf, 50, 0, -np.inf, 50, 1, -np.inf]
        bounds_upper = [np.inf, 200, 50, np.inf, 150, 30, np.inf]
        maxfev=100000
        try:
            fit_y_dg, chi_square_doubleGauss, res_dg, parameters_dg = fit_waveform(DoubleGauss, temp_x, temp_y, guess, maxfev, function_name = "Double Gaussian", parameter_names = ['A', 'x0', 'sigma0', 'B', 'x1', 'sigma1', 'bias'], bounds = [bounds_lower, bounds_upper])
        except RuntimeError:
            print("Warning RuntimeError on Convolution")
            continue

        #Triple Gaussian
        #A, x0, sigma0, B,x1, sigma1, C, x2, sigma2, bias):
        #guess = [10, 80, 1, -400, 100, 15, 10, 140, 5, 2160]
        #bounds_lower = [-np.inf, 50, 0, -np.inf, 0, 1, -np.inf, 100, 0, -np.inf]
        #bounds_upper = [np.inf, 200, 50, np.inf, 200, 30, np.inf, 200, 50, np.inf]
        #maxfev=10000
        #fit_y_tg, chi_square_tripleGauss, res_tg, parameters_tg = fit_waveform(TripleGauss, x, y, guess, maxfev, function_name = "Triple Gaussian", parameter_names = ['A', 'x0', 'sigma0', 'B', 'x1', 'sigma1', 'C', 'x2', 'sigma2', 'bias'], bounds = [bounds_lower, bounds_upper])

        #Convolution
        #A, x0, sigma, bias, tau, x1, b):
        guess = [-10000, 100, 12, 2150, 5, 100, -10000]
        bounds_lower = [-np.inf, 0, 5, -np.inf, 0, 0, -np.inf]
        bounds_upper = [np.inf, 200, 50, np.inf, 50, 100, np.inf]
        maxfev=100000
        fit_y_convol, chi_square_convol, res_convol, parameters_convol = fit_waveform(Convol, temp_x, temp_y, guess, maxfev, function_name = "Convolution", parameter_names = ['A', 'x0', 'sigma0', 'bias', 'tau', 'x1', 'b'], bounds = [bounds_lower, bounds_upper])

        #Double Gauss Convolution
        #A, x0, sigma0, B, x1, sigma1, tau, x2, C, bias):
        #guess = [-10, 180, 5, -5000, 100, 12, 30, 50, -800, 2160]
        #bounds_lower = [-10000, 0, 0, -np.inf, 80, 0, 0, 0, -np.inf, -np.inf]
        #bounds_upper = [10000, 300, 50, np.inf, 120, 50, 100, 300, np.inf, np.inf]
        #maxfev=10000
        #try:
        #    fit_y_dgConvol, chi_square_dgConvol, res_dgConvol, parameters_dgConvol = fit_waveform(DoublegaussConvol, temp_x, temp_y, guess, maxfev, function_name = "DG Convol", parameter_names = ['A', 'x0', 'sigma0', 'B', 'x1', 'sigma1', 'tau', 'x2', 'C', 'bias'], bounds = [bounds_lower, bounds_upper])
        #except RuntimeError:
        #    print("Warning RuntimeError on DG Convolution")
        #    continue

        #a, x0, sigma0, b, sigma1, x1, tau, , x2, bias
        guess = [-5000, 180, 12, -800, 10, 100, 50, 100, 2160]
        bounds_lower = [-np.inf, 0, 0, -np.inf, 0, 0, 0, 0, -np.inf]
        bounds_upper = [np.inf, 300, 100, np.inf, 50, 200, 50, 200, np.inf]
        maxfev=10000
        try:
            fit_y_dgConvol, chi_square_dgConvol, res_dgConvol, parameters_dgConvol = fit_waveform(DoublegaussConvol, temp_x, temp_y, guess, maxfev, function_name = "DG Convol", parameter_names = ['A', 'x0', 'sigma0', 'B', 'sigma1', 'x1', 'tau', 'x2', 'bias'], bounds = [bounds_lower, bounds_upper])
        except RuntimeError:
            print("Warning RuntimeError on DG Convolution")
            continue

        #Double Gauss Convolution + Background
        #(a, x0, sigma0, sigma1, tau0, tau1, bias)
        #guess = [-1000, 100, 15, 15, 15, 15, 2160]
        #bounds_lower = [-np.inf, 80, 0, 0, 0, 0, -np.inf]
        #bounds_upper = [np.inf, 120, 50, 50, 100, 100, np.inf]
        #maxfev=100000
        #try:
        #    fit_y_dgConvolBkg, chi_square_dgConvolBkg, res_dgConvolBkg, parameters_dgConvolBkg = fit_waveform(DoublegaussConvolBackground, temp_x, temp_y, guess, maxfev, function_name = "DG ConvolBkg", parameter_names = ['A', 'x0', 'sigma0', 'sigma1', 'tau0', 'tau1', 'bias'], bounds = [bounds_lower, bounds_upper])
        #except RuntimeError:
        #    print("Warning RuntimeError on DG Convolution + Bkg")
        #    continue

        #B.append(parameters_dgConvol[0])
        #C.append(parameters_dgConvol[3])
        #tau.append(parameters_dgConvol[5])
        #sigma0.append(parameters_dgConvol[2])
        #sigma1.append(parameters_dgConvol[4])

        
        make_waveform(temp_y , temp_x, 'ns', 'CT ADC Value ', 'plots/ct//waveforms/waveform_spill'+str(i)+'_bunch'+str(j)+'.png', fit_val=[fit_y, fit_y_dg, fit_y_convol, fit_y_dgConvol], fit_names = ["Gaussian", "Double Gaussian", "Convolution", "DG Convol"], fit_chi = [chi_square_gauss, chi_square_doubleGauss, chi_square_convol, chi_square_dgConvol], savefig=True)

        print(f"Reduced Chi square gauss: {chi_square_gauss:.3f}, double gauss: {chi_square_doubleGauss:.3f}, double gauss convol: {chi_square_dgConvol:.3f}")


        np.savez('ct_data.npz',B=B, C=C, tau=tau, sigma0=sigma0, sigma1=sigma1) 




    plt.scatter(x, y, label='data')
    plt.legend()
    plt.savefig(f"plots/ct/spill{i}_ct.png")
    plt.close()






