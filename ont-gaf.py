import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GASF, GADF
from tqdm import tqdm
import warnings
import json
import os
warnings.filterwarnings("ignore")
def get_train_data():
    path=r'data\ont-data\train_set_1024'
    x = []
    y = []
    for file in os.listdir(path):
        with open(os.path.join(path, file)) as json_file:
            roi = json.load(json_file)
            x.append(roi['data'])
            y.append(int(roi['label']))
    x=np.array(x)
    y=np.array(y)
    return x,y
def get_test_data():
    path = r'data\ont-data\test_set_CNN'
    x = []
    y = []
    for file in os.listdir(path):
        with open(os.path.join(path, file)) as json_file:
            roi = json.load(json_file)
            x.append(roi['data'])
            y.append(int(roi['label']))
    x = np.array(x)
    y = np.array(y)
    return x, y
# x_data,y_data=get_train_data()
# x_data,y_data=get_test_data()
# array=[]
# for i in tqdm(range(len(x_data))):
#     x = np.array(x_data[i])
#     X = x[0:]
#     X = X.reshape(1, -1)
#     # print(type(X), X.shape)
#     image_size = 64
#     gasf = GASF(image_size)
#     X_gasf = gasf.fit_transform(X)
#     # print(X_gasf.shape)
#     gadf = GADF(image_size)
#     X_gadf = gadf.fit_transform(X)
#     array.append(X_gasf)
#     plt.figure()
#     plt.suptitle('gunpoint_index_' + str(0))
#     ax1 = plt.subplot(121)
#     ax1.plot(np.arange(len(X[0])), X[0])
#     plt.title('rescaled time series')
#     ax2 = plt.subplot(122, polar=True)
#     r = np.array(range(1, len(X[0]) + 1)) / 150
#     theta = np.arccos(np.array(X[0])) * 2 * np.pi  # radian -> Angle
#
#     ax2.plot(theta, r, color='r', linewidth=3)
#     plt.title('polar system')
#     plt.savefig(r".\plot\polar_ont\out_%i.png" % i)
#     plt.close('all')
    # Show the results for the first time series
    # plt.close('all')
    # plt.figure(figsize=(8, 8))
    # plt.plot()
    # plt.imshow(X_gadf[0], cmap='rainbow', origin='lower')
    # # plt.title("GASF", fontsize=16)
    # # #plt.imshow(X_gadf[0], cmap='rainbow', origin='lower')
    # # # plt.title("GADF", fontsize=16)
    # plt.savefig(r"data\ont-data\GADF\plot\test\out_%i.png" % i)
    # plt.close('all')
# save_data = r"data\ont-data\GASF\train\gasf_train.npz"
# np.savez(save_data, np.array(array))
# save_data = r"data\ont-data\GASF\train\label_train.npz"
# np.savez(save_data, np.array(y_data))

f1=open(r'data\ont-data\train_set_CNN\train_1070.json',"r")
x=[]
y=[]
for line in tqdm(f1):
    decodes=json.loads(line)
    x.append(decodes['data'])

f1.close()
x=np.array(x[0])
X = x[0:]
X = X.reshape(1, -1)
    # print(type(X), X.shape)
image_size = 64
gasf = GASF(image_size)
X_gasf = gasf.fit_transform(X)
    # print(X_gasf.shape)
gadf = GADF(image_size)
X_gadf = gadf.fit_transform(X)
plt.figure()
plt.suptitle('gunpoint_index_' + str(0))
ax1 = plt.subplot(121)
ax1.plot(np.arange(len(X[0])), X[0])
plt.title('rescaled time series')
ax2 = plt.subplot(122, polar=True)
r = np.array(range(1, len(X[0]) + 1)) / 150
theta = np.arccos(np.array(X[0])) * 2 * np.pi  # radian -> Angle

ax2.plot(theta, r, color='r', linewidth=3)
plt.title('polar system')
plt.show()

