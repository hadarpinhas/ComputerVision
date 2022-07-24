import numpy as np
import cv2
import matplotlib.pyplot as plt


def predict_evaluation(image, label, pred):
    '''
    '''
    # transform gray image to rgb
    img = np.array(255*image, np.uint8)
    # rgb_img = cv2.cvtColor(255*img, cv2.COLOR_GRAY2RGB)
    # scale pred and mask's pixel range to 0~255
    im_label = np.array(255*label, dtype=np.uint8)
    im_pred = np.array(255*pred, dtype=np.uint8)

    img_pred = cv2.addWeighted(img, 1, im_pred, 0.3, 0)
    img_label = cv2.addWeighted(img, 1, im_label, 0.3, 0)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(img_label)
    plt.title('Ground truth')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(img_pred)
    plt.title('Prediction')
    plt.axis('off')
'''
    # transform both of them to rgb
    rgb_label = cv2.cvtColor(im_label, cv2.COLOR_GRAY2RGB)
    rgb_pred = cv2.cvtColor(im_pred, cv2.COLOR_GRAY2RGB)

    rgb_label[:, :, 1:3] = 0*rgb_label[:, :, 1:2]
    rgb_pred[:, :, 0] = 0*rgb_pred[:, :, 0]
    rgb_pred[:, :, 2] = 0*rgb_pred[:, :, 2]

    img_pred = cv2.addWeighted(rgb_img, 1, rgb_pred, 0.3, 0)
    img_label = cv2.addWeighted(rgb_img, 1, rgb_label, 0.3, 0)

    plt.figure(figsize=(10, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb_img)
    plt.title('Original image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(img_label)
    plt.title('Ground truth')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(img_pred)
    plt.title('Prediction')
    plt.axis('off')
    '''