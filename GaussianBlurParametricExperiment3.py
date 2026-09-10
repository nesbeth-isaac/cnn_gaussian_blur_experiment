"""
Experiment Name: Parametric Experiment 3
Experiment Description: This experiment will build on the results of previously identified optium architectures and
                        determine the kernel size which increases model performance the most.
"""

import matplotlib.pyplot as plt
import numpy as np
import math

from GausianBlurCategoriser.GaussianBlurParametricExperiment1 import CNN_ARCHITECTURE
from UploadToBucket import UploadToBucket
from GausiannBlurGenerator import GausiannBlurGenerator
from DataLoaderFactory import *
from ModelArchitecture import ModelGenerator
from TrainModel import ModelTrainingFactory
from UploadToBucket import UploadToBucket
from GraphFactory import GraphFactory

import configparser

print("-------------------- Configuring Parameters --------------------")

parametric_config = configparser.ConfigParser()
parametric_config.read('parametric_configuration.ini')

EXPERIMENT_NAME = parametric_config["parametric_experiment_2"]["experiment_name"]
COMMON_ATTRIBUTES_ID = parametric_config["common_attributes"]["common_attributes_config_id"]

CNN_START_KERNEL_SIZE = int(parametric_config[EXPERIMENT_NAME]["cnn_start_kernel_size"])
CNN_END_KERNEL_SIZE = int(parametric_config[EXPERIMENT_NAME]["cnn_end_kernel_size"])

CNN_INCREMENT_RATIO = int(parametric_config[EXPERIMENT_NAME]["num_parameters"])

NUM_TRAINING_EPOCHS = int(parametric_config[EXPERIMENT_NAME]["num_training_epochs"])
LEARNING_RATE = float(parametric_config[EXPERIMENT_NAME]["learning_rate"])

EXPERIMENT_PARAMETERS = np.linspace(CNN_START_KERNEL_SIZE, CNN_END_KERNEL_SIZE, CNN_INCREMENT_RATIO)

FFN_ARCHITECTURE = {1: [100, 75], 2: [75, 50], 3: [50, 25], 4: [25, 1]}

NUM_PADDING = 1
STRIDE = 1
NUM_LAYERS = 4

CNN_ARCHITECTURE = {}

#Data Parameters

INPUT_SIZE = int(parametric_config[COMMON_ATTRIBUTES_ID]["input_size"])
IMAGE_SIZE = int(parametric_config[COMMON_ATTRIBUTES_ID]["image_size"])
GAUSSIAN_START_RANGE_X = int(parametric_config[COMMON_ATTRIBUTES_ID]["gaussian_range_start_x"])
GAUSSIAN_END_RANGE_X = int(parametric_config[COMMON_ATTRIBUTES_ID]["gaussian_range_end_x"])
GAUSSIAN_START_RANGE_Y = int(parametric_config[COMMON_ATTRIBUTES_ID]["gaussian_range_start_y"])
GAUSSIAN_END_RANGE_Y = int(parametric_config[COMMON_ATTRIBUTES_ID]["gaussian_range_end_y"])
NUM_IMAGES = int(parametric_config[COMMON_ATTRIBUTES_ID]["num_images"])

#Cloud Config
cloud_config = configparser.ConfigParser()
cloud_config.read('cloud_configuration.ini')

S3_BUCKET_ID = str(cloud_config['s3_bucket_information']['s3_bucket_id'])

upload_to_bucket = UploadToBucket(S3_BUCKET_ID)

print("-------------------- Generating Data --------------------")
gaussian_generator = GausiannBlurGenerator(NUM_IMAGES, IMAGE_SIZE, GAUSSIAN_START_RANGE_X, GAUSSIAN_END_RANGE_X,
                                           GAUSSIAN_START_RANGE_Y, GAUSSIAN_END_RANGE_Y)
synthetic_image_data, labels = gaussian_generator.generate_gausiann_blurs()

train_loader, dev_test_loader, final_test_loader = produce_dataloader(synthetic_image_data, labels)


#Generate CNN architecture

model_store = []
training_store = []