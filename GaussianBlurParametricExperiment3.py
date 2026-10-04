"""
Experiment Name: Parametric Experiment 3
Experiment Description: This experiment will build on the results of previously identified optium architectures and
                        determine the kernel size which increases model performance the most, and will store Feature
                        Activation Maps in S3.
"""

import matplotlib.pyplot as plt
import numpy as np
import math

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

EXPERIMENT_NAME = parametric_config["parametric_experiment_3"]["experiment_name"]
COMMON_ATTRIBUTES_ID = parametric_config["common_attributes"]["common_attributes_config_id"]

CNN_START_KERNEL_SIZE = int(parametric_config[EXPERIMENT_NAME]["cnn_start_kernel_size"])
CNN_END_KERNEL_SIZE = int(parametric_config[EXPERIMENT_NAME]["cnn_end_kernel_size"])

CNN_INCREMENT_RATIO = int(parametric_config[EXPERIMENT_NAME]["num_parameters"])

NUM_TRAINING_EPOCHS = int(parametric_config[EXPERIMENT_NAME]["num_training_epochs"])
LEARNING_RATE = float(parametric_config[EXPERIMENT_NAME]["learning_rate"])

EXPERIMENT_PARAMETERS = np.floor(np.linspace(CNN_START_KERNEL_SIZE, CNN_END_KERNEL_SIZE, CNN_INCREMENT_RATIO))

FFN_ARCHITECTURE = {1: [100, 75], 2: [75, 50], 3: [50, 25], 4: [25, 1]}

NUM_PADDING = 1
STRIDE = 1
NUM_LAYERS = 4

CNN_ARCHITECTURE = {1 : [1, 10, 10, 1, 1], 2: [10, 50, 5, 1, 1]}

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

print("-------------------- Generating Experimental Architectures --------------------")

model_store = []
training_store = []

for experiment in EXPERIMENT_PARAMETERS:
    current_experimental_architecture = CNN_ARCHITECTURE

    model_name = EXPERIMENT_NAME + "_Kernel_" + str(experiment)

    for key in list(current_experimental_architecture.keys()):
        current_experimental_architecture[key][2] = math.floor(experiment)

    model_store.append(ModelGenerator(model_name))
    model_store[-1].generate_cnn_layers_model(current_experimental_architecture, INPUT_SIZE)
    model_store[-1].generate_fnn_layers_models(FFN_ARCHITECTURE)

#Training Models

print("-------------------- Training and Testing Models --------------------")

for model in model_store:
    training_store.append(ModelTrainingFactory(model, LEARNING_RATE, EXPERIMENT_NAME))

    training_store[-1].train_model(train_loader, dev_test_loader, NUM_TRAINING_EPOCHS)

    training_store[-1].set_summary_graph_name()

    training_store[-1].generate_summary_data()
    training_store[-1].generate_feature_map_summary_file()

    upload_to_bucket.upload_file(training_store[-1].get_summary_graph_name(),
                                 training_store[-1].get_summary_graph_name())

    for feature_map_name in training_store[-1].get_feature_map_names():
        upload_to_bucket.upload_file(feature_map_name, feature_map_name)

print("Trained and uploaded all individual summary files. ")
print("Generating Experiment Summary Graph")

graph_factory = GraphFactory(training_store)

graph_factory.set_graph_name(EXPERIMENT_NAME)
graph_name = graph_factory.plot_graph()

upload_to_bucket.upload_file(graph_name, graph_name)

print("Experiment Completed")