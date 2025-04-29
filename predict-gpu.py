# from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from keras.models import Model
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from keras.layers import ReLU
from keras.callbacks import TensorBoard
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras import losses

from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

from train import model, input_train, index_array



# Step size for parameter iteration
step = 0.5

# Arrays to store the output data
out_s11 = np.empty((0, 10))
out_gain = np.empty((0, 10))
out_both = np.empty((0, 10))
out_max = np.empty((0, 10))
out_total = np.empty((0, 10))

# File paths for saving the output data
S11_Save_path = "Output_Bandwidth1000+.txt"
Gain_Save_path = "Output_Gain9.5+.txt"
Both_Save_path = "Output_Both.txt"
MAX_Save_path = "Output_Max.txt"
Total_Save_path = "Output_Total.txt"

# Calculate the total number of iterations based on the parameter ranges and step size



# Initialize counters and variables
num = 0
delta = 1000  # Interval for progress updates
Gain_max = 0  # Maximum gain encountered
BandWidth_max = -1  # Maximum bandwidth encountered

total_num = (5 / 1 + 1) * (12 / step + 1) * (8 / step + 1) * (4 / step + 1) * (5 / step + 1) * (4 / step + 1)
print("Total Number:", total_num)

# Iterate over the parameter space
h_range = np.arange(5,11,1)
Scale_X_range = np.arange(69.0,81.5,0.5)
Scale_Y_range = np.arange(37.0,45.5,0.5)
Offset_y_range = np.arange(-8.0,-3.5,0.5)
Scale_Slot_range = np.arange(8.0,13.5,0.5)
Uw3_range = np.arange(9.0,13.5,0.5)

# total_num = 1
# print("Total Number:", total_num)
#
# h_range = np.arange(7,8,1)
# Scale_X_range = np.arange(71.0,71.5,0.5)
# Scale_Y_range = np.arange(40.0,40.5,0.5)
# Offset_y_range = np.arange(-6.5,-6.0,0.5)
# Scale_Slot_range = np.arange(8.0,8.5,0.5)
# Uw3_range = np.arange(9.0,9.5,0.5)

grids = np.meshgrid(h_range,Scale_X_range,Scale_Y_range,Offset_y_range,Scale_Slot_range,Uw3_range)
params = np.vstack([g.ravel() for g in grids]).T
normalized_params = (params - index_array[0]) / (index_array[1] - index_array[0])
pre_y = model.predict(normalized_params, batch_size=1024)
print(len(pre_y),total_num,len(params))

pre_y[:, 1:] = -70 * pre_y[:, 1:]
pre_y[:, 0] = 5 * pre_y[:, 0] + 5
S11_batches = pre_y[:, 1:]
Gain_batches = pre_y[:, 0]
out_total = np.zeros((int(total_num), 6 + len(params[0])))

print(out_total.shape)
#base = 811000
cnt = 0
for k,S11 in enumerate(S11_batches):#[base:base+1000]
    # Calculate the bandwidth
    S11 = [S11]
    index , Gain = params[k] , Gain_batches[k]
    h = index[0]
    Gain = [Gain]
    index = index.reshape(1,6)
    #print(index)
    result_2p3 = np.empty((0, 9))
    band = [np.where(line <= -10)[0] / 200 + 1.5 for line in S11]
    #print(S11)

    #search for 2.3Ghz
    for i in range(1):
        a, b = [], []
        if band[i].any():
            #print('ok')
            a = [band[i][0]]
            for j in range(1, len(band[i])):
                if round((band[i][j] - band[i][j - 1]) * 1000) / 1000 == 0.005:
                    a.append(band[i][j])
                else:
                    break
            if j < len(band[i]) - 1:
                j0 = j
                b = [band[i][j0]]
                for j in range(j0 + 1, len(band[i])):
                    if round((band[i][j] - band[i][j - 1]) * 1000) / 1000 == 0.005:
                        b.append(band[i][j])
                    else:
                        break
        if not (2.3 in a):
            a = []
        if not (2.3 in b):
            b = []
        # if not(a == []) and int((a[-1] - a[0]) * 1000) / h[i] >= 130:
        if not (a == []):
            # print('ID:', i + 1, ' Band:', a[0], 'GHz -', a[-1], 'GHz  Band Width:', int((a[-1] - a[0]) * 1000), 'MHz  H:', h[i], 'mm  B/H:', int(int((a[-1] - a[0]) * 1000) / h[i]))
            result_2p3 = np.append(result_2p3, [
                np.concatenate((index[i], np.array([a[0], a[-1], (a[-1] - a[0]) * 1000])),
                               axis=0)], axis=0)
            #print(1)
        # if not (b == []) and int((b[-1] - b[0]) * 1000) / h[i] >= 130:
        elif not (b == []):
            # print('ID:', i + 1, ' Band:', b[0], 'GHz -', b[-1], 'GHz  Band Width:', int((b[-1] - b[0]) * 1000), 'MHz  H:', h[i], 'mm  B/H:', int(int((b[-1] - b[0]) * 1000) / h[i]))
            result_2p3 = np.append(result_2p3, [
                np.concatenate((index[i], np.array([b[0], b[-1], (b[-1] - b[0]) * 1000])),
                               axis=0)], axis=0)
            #print(2)
        else:
            result_2p3 = np.append(result_2p3,
                               [np.concatenate((index[i], np.array([0, 0, 0])), axis=0)],
                               axis=0)
    result = np.empty((0,12))
    #search for 3.3Ghz
    for i in range(1):
        a, b = [], []
        if band[i].any():
            #print('ok')
            a = [band[i][0]]
            for j in range(1, len(band[i])):
                if round((band[i][j] - band[i][j - 1]) * 1000) / 1000 == 0.005:
                    a.append(band[i][j])
                else:
                    break
            if j < len(band[i]) - 1:
                j0 = j
                b = [band[i][j0]]
                for j in range(j0 + 1, len(band[i])):
                    if round((band[i][j] - band[i][j - 1]) * 1000) / 1000 == 0.005:
                        b.append(band[i][j])
                    else:
                        break
        if not (3.3 in a):
            a = []
        if not (3.3 in b):
            b = []
        # if not(a == []) and int((a[-1] - a[0]) * 1000) / h[i] >= 130:
        if not (a == []):
            # print('ID:', i + 1, ' Band:', a[0], 'GHz -', a[-1], 'GHz  Band Width:', int((a[-1] - a[0]) * 1000), 'MHz  H:', h[i], 'mm  B/H:', int(int((a[-1] - a[0]) * 1000) / h[i]))
            result = np.append(result, [
                np.concatenate((result_2p3[0], np.array([a[0], a[-1], (a[-1] - a[0]) * 1000])),
                               axis=0)], axis=0)
            #print(1)
        # if not (b == []) and int((b[-1] - b[0]) * 1000) / h[i] >= 130:
        elif not (b == []):
            # print('ID:', i + 1, ' Band:', b[0], 'GHz -', b[-1], 'GHz  Band Width:', int((b[-1] - b[0]) * 1000), 'MHz  H:', h[i], 'mm  B/H:', int(int((b[-1] - b[0]) * 1000) / h[i]))
            result = np.append(result, [
                np.concatenate((result_2p3[0], np.array([b[0], b[-1], (b[-1] - b[0]) * 1000])),
                               axis=0)], axis=0)
            #print(2)
        else:
            result = np.append(result,
                               [np.concatenate((result_2p3[0], np.array([0, 0, 0])), axis=0)],
                               axis=0)

    if result[0,-1] >=50 and result[0,-4] >= 50:
        out_total[cnt] = result
        cnt += 1

    #Append the current result to the total output array
    # out_total[k] = result
    #
    # # Check if the bandwidth is greater than 1000 MHz
    # if result[0, -2] > 1000:
    #     # Append the result to the S11 output array
    #     out_s11 = np.append(out_s11, result,
    #                         axis=0)
    #
    # # Check if the gain is greater than 9.5
    #
    # if Gain[0] > 9.5:
    #     # Append the result to the gain output array
    #     out_gain = np.append(out_gain, result,
    #                          axis=0)
    #
    # # Check for specific conditions and append to the both output array
    # if h <= 7 and result[0, -2] > 700 and Gain[0] > 9.0:
    #     out_both = np.append(out_both, result,
    #                          axis=0)
    #     #print('ook')
    #     # Update the maximum gain and bandwidth encountered
    #     if Gain[0] > Gain_max:
    #             Gain_max = Gain[0]
    #             arg_gain = result[0]
    #
    #     if result[0, -2] > BandWidth_max:
    #             BandWidth_max = result[0, -2]
    #             arg_bandwidth = result[0]

    if k % delta == 0:
                            print("%d / %d is Done!" % (k, total_num))
                            print("Current Parameters:", index[0])
                            print()
#out_total = out_total[:1000]


#Save the output data to text files
# np.savetxt(S11_Save_path, out_s11, fmt='%.3f')
# np.savetxt(Gain_Save_path, out_gain, fmt='%.3f')
# np.savetxt(Both_Save_path, out_both, fmt='%.3f')
#
# # Append the maximum gain and bandwidth results to the out_max array
# out_max = np.append(out_max, [arg_gain], axis=0)
# out_max = np.append(out_max, [arg_bandwidth], axis=0)
#
# # Save the out_max array
# np.savetxt(MAX_Save_path, out_max, fmt='%.3f')
out_total = out_total[:cnt]

# Save the total output array
np.savetxt(Total_Save_path, out_total, fmt='%.3f')



# Update the iteration counter


# # Calculate the bandwidth
# #
# index = np.array([7.0,71.0,40.0,-6.5,8.0,9.0]).reshape([1,6])
# index_norm = (index - index_array[0, :]) / (index_array[1, :] - index_array[0, :])
# pre_y = model.predict(index_norm)
# pre_y[:, 1:] = -70 * pre_y[:, 1:]
# pre_y[:, 0] = 5 * pre_y[:, 0] + 5
# S11 = pre_y[:,1:]
# Gain = pre_y[:, 0]
# #print(S11)
# result = np.empty((0, 10))
# band = [np.where(line <= -10)[0] / 200 + 1.5 for line in S11]
# for i in range(1):
#     a, b = [], []
#     if band[i].any():
#         a = [band[i][0]]
#         for j in range(1, len(band[i])):
#             if round((band[i][j] - band[i][j - 1]) * 1000) / 1000 == 0.005:
#                 a.append(band[i][j])
#             else:
#                 break
#         if j < len(band[i]) - 1:
#             j0 = j
#             b = [band[i][j0]]
#             for j in range(j0 + 1, len(band[i])):
#                 if round((band[i][j] - band[i][j - 1]) * 1000) / 1000 == 0.005:
#                     b.append(band[i][j])
#                 else:
#                     break
#     if not (2.45 in a):
#         a = []
#     if not (2.45 in b):
#         b = []
#     # if not(a == []) and int((a[-1] - a[0]) * 1000) / h[i] >= 130:
#     if not (a == []):
#         # print('ID:', i + 1, ' Band:', a[0], 'GHz -', a[-1], 'GHz  Band Width:', int((a[-1] - a[0]) * 1000), 'MHz  H:', h[i], 'mm  B/H:', int(int((a[-1] - a[0]) * 1000) / h[i]))
#         result = np.append(result, [
#             np.concatenate((index[i], np.array([a[0], a[-1], (a[-1] - a[0]) * 1000]),Gain),
#                            axis=0)], axis=0)
#     # if not (b == []) and int((b[-1] - b[0]) * 1000) / h[i] >= 130:
#     elif not (b == []):
#         # print('ID:', i + 1, ' Band:', b[0], 'GHz -', b[-1], 'GHz  Band Width:', int((b[-1] - b[0]) * 1000), 'MHz  H:', h[i], 'mm  B/H:', int(int((b[-1] - b[0]) * 1000) / h[i]))
#         result = np.append(result, [
#             np.concatenate((index[i], np.array([b[0], b[-1], (b[-1] - b[0]) * 1000]),Gain),
#                            axis=0)], axis=0)
#     else:
#         result = np.append(result,
#                            [np.concatenate((index[i], np.array([0, 0, 0]),Gain), axis=0)],
#                           axis=0)
#
#     #Append the current result to the total output array
#
# h = index[0][0]
#
# # Check if the bandwidth is greater than 1000 MHz
# print( result[0, -2]  ,result[0, -2] > 1000)
#
#
# # Check if the gain is greater than 9.5
#
# print(Gain[0],Gain[0] > 9.5)
#     # Append the result to the gain output array
#
#
# # Check for specific conditions and append to the both output array
# print( h, h <= 7 and result[0, -2] > 700 and Gain[0] > 9.0)
#
# if Gain[0] > Gain_max:
#         Gain_max = Gain[0]
#         arg_gain = result[0]
#
# if result[0, -2] > BandWidth_max:
#         BandWidth_max = result[0, -2]
#         arg_bandwidth = result[0]
#
# print(result)
# #


