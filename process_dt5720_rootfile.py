#Converts root file created by oneDigitizer_midas2root.exe to hdf5
#First argument should be name of rootfile

import sys

import uproot
import h5py

args = sys.argv[1:]

data1 = uproot.open(args[0]+':midas_data1')
ch0 = data1['Channel0_arr'].array(library='np')
ch1 = data1['Channel1_arr'].array(library='np')
ch2 = data1['Channel2_arr'].array(library='np')
ch3 = data1['Channel3_arr'].array(library='np')
timestamp = data1['timestamp'].array(library='np')
triggerTime = data1['triggerTime'].array(library='np')

#Get filename
filename = args[0].split('/')[-1]
filename_no_dot = filename.split('.')[0]
hdf5_filename = '/home/t2k_otr/data_hdf5/'+filename_no_dot + '.h5'

hf = h5py.File(hdf5_filename,'w')
hf.create_dataset('channel0', data=ch0)
hf.create_dataset('channel1', data=ch1)
hf.create_dataset('channel2', data=ch2)
hf.create_dataset('channel3', data=ch3)
hf.create_dataset('timestamp', data=timestamp)
hf.create_dataset('triggerTime', data=triggerTime)
hf.close()

