## imports
import math
import matplotlib.pyplot as plt
import numpy as np

data_set = np.genfromtxt("hw03_data_set.csv", delimiter = ",", skip_header=True)

y_training = data_set[:150,1].astype(int)
y_test = data_set[150:,1].astype(int)

x_training = data_set[:150,0]
x_test = data_set[150:,0]

## parameters 
K = np.max(y_training)
N = data_set.shape[0]

minimum_value = min(x_training)
maximum_value = max(x_training)

bin_width = 0.37
origin = 1.5
data_points = math.ceil((maximum_value - origin) * 100 + 1)
data_interval = np.linspace(origin, maximum_value, data_points)

## borders in the regressogram
left_borders = np.arange(origin, maximum_value, bin_width)
right_borders = np.arange(origin + bin_width, maximum_value + bin_width, 
                          bin_width)
## regressogram score values array
reg_res = np.asarray([np.sum(((left_borders[i] < x_training) & (x_training <= 
            right_borders[i]))*y_training) / np.sum((left_borders[i] < x_training) 
            & (x_training <= right_borders[i])) for i in range(len(left_borders))])
## regressogram plot                                               
plt.figure(figsize = (10, 4))
for i in range(len(left_borders)):
    plt.plot([left_borders[i], right_borders[i]], [reg_res[i], reg_res[i]], "k-")
for i in range(len(left_borders) - 1):
    plt.plot([right_borders[i], right_borders[i]], [reg_res[i], reg_res[i + 1]], "k-")   
plt.plot(x_training,y_training,"b.", markersize = 8,label="training", alpha = 0.3)
plt.plot(x_test,y_test,"r.", markersize = 8,label="test")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption(min)")
plt.title("h=0.37")
plt.legend(loc='upper left')
plt.show()

## regressogram rmse calculation
rmse_reg = 0
for i in range(len(y_test)):
    for j in range(len(right_borders)):
        if(x_test[i] < right_borders[j] and x_test[i] >= left_borders[j]):
            rmse_reg += (y_test[i] - reg_res[j]) ** 2
rmse_reg = math.sqrt(rmse_reg / len(y_test))
print("Regressogram => RMSE is {:.4f} when h is 0.37".format(rmse_reg))

## RMS score values array
rms_res = np.stack([np.sum((((x - 0.5 * bin_width) < x_training) & (x_training <= 
                (x + 0.5 * bin_width))) * y_training) / np.sum(((x - 0.5 * bin_width) < 
                x_training) & (x_training <= (x + 0.5 * bin_width))) for x in data_interval])
## RMS score values array for values in x_test                                                           
rms_xtest_res = np.stack([np.sum((((x - 0.5 * bin_width) < x_training) & (x_training <= 
                (x + 0.5 * bin_width)))*y_training) / np.sum(((x - 0.5 * bin_width) < 
                x_training) & (x_training <= (x + 0.5 * bin_width))) for x in x_test])   
## RMS plot                                                             
plt.figure(figsize = (10, 4))
plt.plot(x_training,y_training,"b.", markersize = 8,label="training", alpha = 0.3)
plt.plot(x_test,y_test,"r.", markersize = 8,label="test") 
plt.plot(data_interval, rms_res, "k-")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption(min)")
plt.title("h=0.37")
plt.legend(loc='upper left')
plt.show()

## RMS rmse calculation
rmse_rms = 0
diff = data_interval[1] - data_interval[0]
for i in range(len(y_test)):
    rmse_rms += (y_test[i] - rms_xtest_res[i]) ** 2
rmse_rms = math.sqrt(rmse_rms / len(y_test))
print("Running Mean Smoother => RMSE is {:.4f} when h is 0.37".format(rmse_rms))

## kernel function
def k(u):
    return np.exp(-u**2/2)/np.sqrt(2*math.pi)

## kernel smoother score values array
kernel_res = np.stack([np.sum(k((data_interval[i] - x_training)/bin_width)*y_training) / np.sum(k((data_interval[i] - x_training)/bin_width)) for i in range(len(data_interval))])
## kernel smoother score values array for values in x_test
kernel_xtest_res = np.stack([np.sum(k((x_test[i] - x_training)/bin_width)*y_training) / np.sum(k((x_test[i] - x_training)/bin_width)) for i in range(len(x_test))])

## kernel smoother plot
plt.figure(figsize = (10, 4))
plt.plot(x_training,y_training,"b.", markersize = 8,label="training", alpha = 0.3)
plt.plot(x_test,y_test,"r.", markersize = 8,label="test")
plt.plot(data_interval, kernel_res, "k-")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption(min)")
plt.title("h=0.37")
plt.legend(loc='upper left')
plt.show()

## kernel smoother rmse calculation
rmse_kernel = 0
for i in range(len(y_test)):
    rmse_kernel += (y_test[i] - kernel_xtest_res[i]) ** 2
rmse_kernel = math.sqrt(rmse_kernel / len(y_test))
print("Kernel Smoother => RMSE is {:.4f} when h is 0.37".format(rmse_kernel))