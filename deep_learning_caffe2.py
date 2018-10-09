CAFFE2_ROOT = "~/.virtualenvs/facecourse-py3/lib/python3.5/site-packages"
CAFFE_MODELS = "~/.virtualenvs/facecourse-py3/lib/python3.5/site-packages/caffe2/python/models"

from caffe2.proto import caffe2_pb2
import numpy as np
import skimage.io
import skimage.transform
import matplotlib.pyplot as mplot
mplot.switch_backend("TKAgg")
import os
from caffe2.python import core, workspace
import urllib
import operator
import numbers
import string

print("MODULES ARE HERE JORDANE!")
IMAGE_LOCATION = "Green_Ball.jpg"
MODEL = 'squeezenet', 'init_net.pb', 'predict_net.pb', 'ilsvrc_2012_mean.npy', 300
codes =  "https://raw.githubusercontent.com/JordaneKM/opencv_caffe2/master/Mapping_AlexNet"

# set paths and variables from model choice and prep image
CAFFE2_ROOT = os.path.expanduser(CAFFE2_ROOT)
CAFFE_MODELS = os.path.expanduser(CAFFE_MODELS)

# mean can be 128 or custom based on the model
# gives better results to remove the colors found in all of the training images
MEAN_FILE = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[3])

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def rescale(img, input_height, input_width):
    print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
    print("Model's input shape is %d x %d" % (input_height, input_width) )
    aspect = img.shape[1]/float(img.shape[0])
    print("Orginal aspect ratio: " + str(aspect))
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    mplot.figure()
    mplot.imshow(imgScaled)
    mplot.axis('on')
    mplot.title('Rescaled image')
    print("New image shape:" + str(imgScaled.shape) + " in HWC")

    return imgScaled

def format_image(frame):
    INPUT_IMAGE_SIZE = MODEL[4]
    # img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
    img = skimage.img_as_float(frame).astype(np.float32)
    img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
    img = crop_center(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)

    return img

def make_pred(img):

    if not os.path.exists(MEAN_FILE):
        mean = 128
    else:
        mean = np.load(MEAN_FILE).mean(1).mean(1)
        mean = mean[:, np.newaxis, np.newaxis]

    # switch to BGR
    img = img[(2, 1, 0), :, :]

    # remove mean for better results
    img = img * 255 - mean

    # add batch size
    img = img[np.newaxis, :, :, :].astype(np.float32)

    # initialize the neural net
    with open(INIT_NET, 'rb') as f:
        init_net = f.read()
    with open(PREDICT_NET, 'rb') as f:
        predict_net = f.read()

    p = workspace.Predictor(init_net, predict_net)

    # run the net and return prediction
    results = p.run({'data': img})

    # turn it into something we can play with and examine which is in a multi-dimensional array
    results = np.asarray(results)

    # Quick way to get the top-1 prediction result
    # Squeeze out the unnecessary axis. This returns a 1-D array of length 1000
    preds = np.squeeze(results)
    # Get the prediction and the confidence by finding the maximum value and index of maximum value in preds array
    curr_pred, curr_conf = max(enumerate(preds), key=operator.itemgetter(1))

    results = np.delete(results, 1)
    index = 0
    highest = 0
    arr = np.empty((0,2), dtype=object)
    arr[:,0] = int(10)
    arr[:,1:] = float(10)
    for i, r in enumerate(results):
        # imagenet index begins with 1!
        i=i+1
        arr = np.append(arr, np.array([[i,r]]), axis=0)
        if (r > highest):
            highest = r
            index = i

    # top N results
    N = 5
    topN = sorted(arr, key=lambda x: x[1], reverse=True)[:N]
    print("Raw top {} results: {}".format(N,topN))

    # Isolate the indexes of the top-N most likely classes
    topN_inds = [int(x[0]) for x in topN]

    return (curr_conf, curr_pred, topN_inds)


def top_five_pred(img):

    with urllib.request.urlopen(codes) as url:

    response = response.decode()
    response = response[:-3]

    list_new = []
    list_new_2 =[]
    n=0
    ind=0

    for line in range(len(response)):
        if response[line].isdigit() and n<10:
            locate = response.find("',",ind,len(response))
            list_new.append(response[line:locate])
            ind = locate+1
            n=n+1
        if response[line].isdigit() and response[line+1].isdigit() and n<100:
            locate = response.find("',",ind,len(response))
            list_new.append(response[line:locate])
            ind = locate+1
            n=n+1
        if response[line].isdigit() and response[line+1].isdigit() and response[line+2].isdigit() and n<1000:
            locate = response.find("',",ind,len(response))
            list_new.append(response[line:locate])
            ind = locate+1
            n=n+1

    result_1,result_2,result_3 = make_pred(img)

    for i in result_3:
        list_new_2.append(list_new[i])

    return list_new_2
