# import wget
# import zipfile
# import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Plot import plot_ellipse_seg_test
from DataIO import load_images_masks
import os
from sklearn.model_selection import train_test_split
from myModel import build_small_unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from tensorflow.keras.optimizers import Adam
import pickle
import keras
from predictEvaluation import predict_evaluation

## data label format
# [filename] \t \n [semi-major axis] \t [semi-minor axis] \t [rotation angle]
#  \t [x-position of the centre of the ellipsoid] \t [y-position of the centre of the ellipsoid] \n [filename] \t

## download data and exctract
'''
# url = 'https://resources.mpi-inf.mpg.de/conference/dagm/2007/Class1_def.zip'
# filename = wget.download(url)
# with zipfile.ZipFile("Class1_def.zip", 'r') as zip_ref:
#     zip_ref.extractall()
'''

## check a single image and label

data_dir = "Class1_def/"
img_path = (os.path.join(data_dir, "1.png"))
plot_ellipse_seg_test(img_path)
# plt.imshow(mpimg.imread(img_path), cmap='gray')
plt.show()


## load and check data
data_dir = "Class1_def"
X, y = load_images_masks(data_dir, img_type='png', img_format='gray', resize=(512, 512), ellipse=True)
print(X.shape,'\n',y.shape)
plt.imshow(X[0,:,:,0], cmap='gray')
# plt.show()
plt.imshow(y[0,:,:,0], cmap='gray')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train.shape
X_test.shape

## Unet

# img_rows = 512
# img_cols = 512
# model = build_small_unet(img_rows, img_cols)
# model.summary()
# model.compile(optimizer=Adam(learning_rate=1e-4) , loss=bce_jaccard_loss, metrics=[iou_score])
# history = model.fit(X_train, y_train, batch_size=10, epochs=50, verbose=1, validation_split=0.1)
# with open('trainHistoryDict', 'wb') as file_pi:
        # pickle.dump(history.history, file_pi)
# keras.models.save_model(model, 'my_model/my_model.h5')

history = pickle.load(open('trainHistoryDict', "rb"))

## plot
# history = model.history.history

plt.figure(figsize=(20, 5))
plt.plot(history['loss'], label='Train loss')
plt.plot(history['val_loss'], label='Val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure(figsize=(20, 5))
plt.plot(history['iou_score'], label='Train IOU')
plt.plot(history['val_iou_score'], label='Val IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()

model = keras.models.load_model('my_model/my_model.h5', compile=False)

predict = model.predict(X_test)

predict_evaluation(X_test[0,:,:,0], y_test[0,:,:,0], predict[0,:,:,0])
plt.show()

