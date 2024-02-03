import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import tensorflow as tf
import os 
import glob
import train 
import metrics 
import model
from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope
from PIL import Image
import cv2
os.getcwd()

def predict_img(path,model,savedir = './predictions/',show = True,savefig = True):
    '''
    Predict your image with trained model and save the prediction
    path --- path of image to predict
    model --- your model
    savedir ---where you save your prediction, default = './predictions/'
    show --- boolean option display your prediction, default = True
    savefig --- boolean option save your prediction, default = True
    Return y_pred (tensor array) shape [1,H,W,1]
    '''
    img_pred = train.read_image(path)
    img_name = path.split('/')[-1]
    img_name = img_name.split('.')[0]
    img_name = img_name + '_predicted'
    y_pred = model.predict(tf.expand_dims(img_pred,axis =0))
    #image_org = Image.open(path)
    image_org = cv2.imread(path, cv2.IMREAD_COLOR)
    image_org = cv2.resize(image_org,(512, 512), interpolation = cv2.INTER_CUBIC)
    image_org = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)

    if show:
        plt.figure(figsize=(10,10))

        # Tạo subplot với 1 hàng và 2 cột
        plt.subplot(1, 2, 1)
        plt.imshow(image_org)
        plt.axis('off')
        plt.title('Image Original')

        plt.subplot(1, 2, 2)
        plt.imshow(y_pred[0][:,:,0])
        plt.axis('off')
        plt.title('Image Result')

        #plt.show()
        # there is no image to save if you don't show anything
        if savefig:
            plt.savefig(savedir + '/resize_img/' + img_name + '.png')
    return y_pred

if __name__ == '__main__':

    
    folder_test = './img/images_Q/'
    for i, filename in enumerate(os.listdir(folder_test)):
    # pred_path = input('Enter your image path: ')
        pred_path = folder_test + filename
        # create results floder
        train.create_dir('./predictions')
        # Set custom classes or functions in model with your custom definition
        with CustomObjectScope({'iou':metrics.iou,'dice_coef':metrics.dice_coef,'dice_loss':metrics.dice_loss}):
        # model = tf.keras.models.load_model('/content/conbtent/MyDrive/model_lung_unet1.h5')
            model = tf.keras.models.load_model('./files/model_lung_unet1.h5')
        # predict
        predict_img(pred_path,model)