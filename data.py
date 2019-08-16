import numpy as np
from PIL import Image
from scipy import misc
import glob

def load_images(globpath, num):
    for i, image in enumerate(globpath):
        with open(image, 'rb') as file:
            img = Image.open(file)
            img = img.resize((48, 48))
            np_img = np.array(img)
            my_list.append(np_img)
        labels.append(num)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y].T
    return Y

my_list = []
labels = []
load_images(glob.glob("senz3d_dataset/acquisitions/S1/G1/*.png"), 0)
load_images(glob.glob("senz3d_dataset/acquisitions/S2/G1/*.png"), 0)
load_images(glob.glob("senz3d_dataset/acquisitions/S3/G1/*.png"), 0)
load_images(glob.glob("senz3d_dataset/acquisitions/S4/G1/*.png"), 0)

load_images(glob.glob("senz3d_dataset/acquisitions/S1/G2/*.png"), 1)
load_images(glob.glob("senz3d_dataset/acquisitions/S2/G2/*.png"), 1)
load_images(glob.glob("senz3d_dataset/acquisitions/S3/G2/*.png"), 1)
load_images(glob.glob("senz3d_dataset/acquisitions/S4/G2/*.png"), 1)

load_images(glob.glob("senz3d_dataset/acquisitions/S1/G3/*.png"), 2)
load_images(glob.glob("senz3d_dataset/acquisitions/S2/G3/*.png"), 2)
load_images(glob.glob("senz3d_dataset/acquisitions/S3/G3/*.png"), 2)
load_images(glob.glob("senz3d_dataset/acquisitions/S4/G3/*.png"), 2)

load_images(glob.glob("senz3d_dataset/acquisitions/S1/G4/*.png"), 3)
load_images(glob.glob("senz3d_dataset/acquisitions/S2/G4/*.png"), 3)
load_images(glob.glob("senz3d_dataset/acquisitions/S3/G4/*.png"), 3)
load_images(glob.glob("senz3d_dataset/acquisitions/S4/G4/*.png"), 3)

load_images(glob.glob("senz3d_dataset/acquisitions/S1/G5/*.png"), 4)
load_images(glob.glob("senz3d_dataset/acquisitions/S2/G5/*.png"), 4)
load_images(glob.glob("senz3d_dataset/acquisitions/S3/G5/*.png"), 4)
load_images(glob.glob("senz3d_dataset/acquisitions/S4/G5/*.png"), 4)

load_images(glob.glob("senz3d_dataset/acquisitions/S1/G6/*.png"), 5)
load_images(glob.glob("senz3d_dataset/acquisitions/S2/G6/*.png"), 5)
load_images(glob.glob("senz3d_dataset/acquisitions/S3/G6/*.png"), 5)
load_images(glob.glob("senz3d_dataset/acquisitions/S4/G6/*.png"), 5)

load_images(glob.glob("senz3d_dataset/acquisitions/S1/G7/*.png"), 6)
load_images(glob.glob("senz3d_dataset/acquisitions/S2/G7/*.png"), 6)
load_images(glob.glob("senz3d_dataset/acquisitions/S3/G7/*.png"), 6)
load_images(glob.glob("senz3d_dataset/acquisitions/S4/G7/*.png"), 6)

load_images(glob.glob("senz3d_dataset/acquisitions/S1/G8/*.png"), 7)
load_images(glob.glob("senz3d_dataset/acquisitions/S2/G8/*.png"), 7)
load_images(glob.glob("senz3d_dataset/acquisitions/S3/G8/*.png"), 7)
load_images(glob.glob("senz3d_dataset/acquisitions/S4/G8/*.png"), 7)

load_images(glob.glob("senz3d_dataset/acquisitions/S1/G9/*.png"), 8)
load_images(glob.glob("senz3d_dataset/acquisitions/S2/G9/*.png"), 8)
load_images(glob.glob("senz3d_dataset/acquisitions/S3/G9/*.png"), 8)
load_images(glob.glob("senz3d_dataset/acquisitions/S4/G9/*.png"), 8)

load_images(glob.glob("senz3d_dataset/acquisitions/S1/G10/*.png"), 9)
load_images(glob.glob("senz3d_dataset/acquisitions/S2/G10/*.png"), 9)
load_images(glob.glob("senz3d_dataset/acquisitions/S3/G10/*.png"), 9)
load_images(glob.glob("senz3d_dataset/acquisitions/S4/G10/*.png"), 9)

load_images(glob.glob("senz3d_dataset/acquisitions/S1/G11/*.png"), 10)
load_images(glob.glob("senz3d_dataset/acquisitions/S2/G11/*.png"), 10)
load_images(glob.glob("senz3d_dataset/acquisitions/S3/G11/*.png"), 10)
load_images(glob.glob("senz3d_dataset/acquisitions/S4/G11/*.png"), 10)

my_list = np.array(my_list)
print('my_list.shape: ', my_list.shape)
labels = np.array(labels)
print("labels.shape ",labels.shape)


p = np.random.permutation(len(my_list))
X_orig = my_list[p]
Y_orig = labels[p]

length = int(len(my_list) * 0.05)
X_test_orig = X_orig[0:length]
Y_test_orig = Y_orig[0:length]

X_dev_orig = X_orig[(length+1):(length*2)]
Y_dev_orig = Y_orig[(length+1):(length*2)]

X_train_orig = X_orig[(length*2+1):len(labels)]
Y_train_orig = Y_orig[(length*2+1):len(labels)]

X_train = X_train_orig/255
X_dev = X_dev_orig/255
X_test = X_test_orig/255
Y_train = convert_to_one_hot(Y_train_orig, 11).T
Y_test = convert_to_one_hot(Y_test_orig, 11).T
Y_dev = convert_to_one_hot(Y_dev_orig, 11).T
def get_datasets():
	return X_train, X_dev, X_test, Y_train, Y_dev, Y_test