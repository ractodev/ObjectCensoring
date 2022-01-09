# Required libraries
import os
from os import path
from io import BytesIO
import tarfile
import tempfile
import cv2
from six.moves import urllib
from copy import deepcopy
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from IPython.display import Image as IMG
import tensorflow as tf

IDENTIFIED_OBJECTS = ""

class Model(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    # Load pretrained model
    def __init__(self, tarball_path):
        self.graph = tf.Graph()
        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    # Function that takes a single image as input and returns resized image and segmentation map
    def run(self, image):
        width, height = image.size
        resize_ratio = 1.0 * (self.INPUT_SIZE / max(width, height))
        target_size = (int(resize_ratio * width), int(resize_ratio * height))

        resized_image = image.convert('RGB').resize(
            target_size, Image.ANTIALIAS)
        batch_segmentation_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        segmentation_map = batch_segmentation_map[0]
        return resized_image, segmentation_map

# Function to colorize detected objects in segmentation map             
def colorize_detected_objects():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap

# Function to label detected objects in segmentation map                
def label_detected_objects(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = colorize_detected_objects()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


# Function to display the process of the segmentation map             
def visualize_segmentation_process(image, segmentation_map):
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    # Display inputted image
    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('INPUT IMAGE')

    # Display segmentation map
    plt.subplot(grid_spec[1])
    segmentation_image = label_detected_objects(segmentation_map).astype(np.uint8)
    plt.imshow(segmentation_image)
    plt.axis('off')
    plt.title('SEGMENTATION MAP')

    # Display overlay view
    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(segmentation_image, alpha=0.7)
    plt.axis('off')
    plt.title('SEGMENTATION OVERLAY')

    # Display meaning behind colors of detected objects
    unique_labels = np.unique(segmentation_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), COCO_OBJECTS[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()


COCO_OBJECTS = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])
FULL_LABEL_MAP = np.arange(len(COCO_OBJECTS)).reshape(len(COCO_OBJECTS), 1)
FULL_COLOR_MAP = label_detected_objects(FULL_LABEL_MAP)


# Download pretrained model from Tensorflow
MODEL_NAME = 'mobilenetv2_coco_voctrainaug'
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}
_TARBALL_NAME = 'deeplab_model.tar.gz'
model_dir = tempfile.mkdtemp()
tf.gfile.MakeDirs(model_dir)
download_path = os.path.join(model_dir, _TARBALL_NAME)
urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX +
                           _MODEL_URLS[MODEL_NAME], download_path)
MODEL = Model(download_path)


# Function to run the segmentation process visualization
def run_segmentation_visualization():
    try:
        original_image = Image.open(IMAGE_NAME)
    except IOError:
        print('Cannot retrieve image. Please check url: ' + url)
        return

    resized_image, segmentation_map = MODEL.run(original_image)
    visualize_segmentation_process(resized_image, segmentation_map)
    return resized_image, segmentation_map


# Function to allow user to choose input image
def choose_input_image():
    while True:
        print("Choose input image: (exclude .jpg)")
        user_input = input() + ".jpg"
        if path.exists(user_input):
            print("You chose", user_input)
            break
        else:
            print("The image name you provided was invalid. Please try again. ") 
    return user_input


# Function to let user decide on what object to hide in picture
def choose_object_type():
    while True:
        print("Choose an object you want to blur if it exists in the picture: \n")
        print(COCO_OBJECTS)
        user_input = input()
        if user_input in COCO_OBJECTS:
            break
        else:
            print("Incorrect input. Please enter one of the categories above: ")
    
    return np.where(COCO_OBJECTS == user_input)


IMAGE_NAME = choose_input_image()
resized_image, segmentation_map = run_segmentation_visualization()

# Convert resized- and original image to numpy array
numpy_image = np.array(resized_image)
original_image = Image.open(IMAGE_NAME)
original_image = np.array(original_image)

# Mask detected object 
object_type = choose_object_type()
object_mapping = deepcopy(numpy_image)
object_mapping[segmentation_map == object_type] = 0
object_mapping[segmentation_map != object_type] = 255

# Resize image to its original size
mapping_resized = cv2.resize(object_mapping, (original_image.shape[1], original_image.shape[0]), Image.ANTIALIAS)
np.unique(mapping_resized)

# Apply Gaussian blur to resized image based on Otsu's Binarization
gray_scale = cv2.cvtColor(mapping_resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_scale, (15, 15), 0)
ret3, thresholded_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
mapping = cv2.cvtColor(thresholded_img, cv2.COLOR_GRAY2RGB)
np.unique(mapping)

# Apply Guassian blur to detected object and leave the background untouched
blurred_original_image = cv2.GaussianBlur(original_image, (251, 251), 0)
layered_image = np.where(mapping != (0, 0, 0), original_image, blurred_original_image)

# Save result
image_rgb = cv2.cvtColor(layered_image, cv2.COLOR_BGR2RGB)
cv2.imwrite("result.jpg", image_rgb)