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

from fit.fit_utilities import move_x_to_peak, average, make_waveform, make_ct_like_otr, fit_waveform, global_fit
from fit.fit_functions import Gauss, DoubleGauss, TripleGauss, Convol, DoublegaussConvol, DoublegaussConvolBackground, convolve_custom 


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

data_x = []
data_y = []

for i, rootfile in enumerate(rootfiles):
    data = uproot.open(rootfile+':g_ch1')
    #To ns
    print(f"Spill {i}")
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

            data_y.append(avg_y)
            data_x.append(avg_x)

            temp_x = avg_x
            temp_y = avg_y
            plt.scatter(temp_x, temp_y, label='data')
            plt.legend()
            plt.savefig(f"plots/ct/average_spill{i}_bunch{j}_ct.png")
            plt.close()
            continue
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

length = min(map(len,data_x))
data_x=np.array([(xi[0:length]) for xi in data_x])
print(data_x.shape)
data_y=np.array([(yi[0:length]) for yi in data_y])
print(data_y.shape)

#Gaussian
#Amplitude, x0, sigma, bias
result = global_fit(DoubleGauss, data_x, data_y)

for tmp_x, tmp_y in zip(data_x, data_y):
    pass






