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

"""
This experiment will evaluate how the number of feature maps impact the model's performance. 
We will use a FFN architecture that performed well previously. We will use 4 CNN convolutional layers. """

print("-------------------- Configuring Parameters --------------------")

parametric_config = configparser.ConfigParser()
parametric_config.read('parametric_configuration.ini')

EXPERIMENT_NAME = parametric_config["parametric_experiment_2"]["experiment_name"]
COMMON_ATTRIBUTES_ID = parametric_config["common_attributes"]["common_attributes_config_id"]

CNN_OUTPUT_FEATURE_MAPS_START_RATIO = int(parametric_config[EXPERIMENT_NAME]["cnn_output_feature_maps_start_ratio"])
CNN_OUTPUT_FEATURE_MAPS_END_RATIO = int(parametric_config[EXPERIMENT_NAME]["cnn_output_feature_maps_end_ratio"])
CNN_INCREMENT_RATIO = int(parametric_config[EXPERIMENT_NAME]["num_parameters"])

NUM_TRAINING_EPOCHS = int(parametric_config[EXPERIMENT_NAME]["num_training_epochs"])
LEARNING_RATE = float(parametric_config[EXPERIMENT_NAME]["learning_rate"])

CNN_PARAMETERS = np.linspace(CNN_OUTPUT_FEATURE_MAPS_START_RATIO,
                             CNN_OUTPUT_FEATURE_MAPS_END_RATIO,
                             CNN_INCREMENT_RATIO)

FFN_ARCHITECTURE = {1: [100, 75], 2: [75, 50], 3: [50, 25], 4: [25, 1]}

KERNEL_SIZE = 1
NUM_PADDING = 1
STRIDE = 1
NUM_LAYERS = 4

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

print("-------------------- Generating Untrained Models --------------------")

for parameter in CNN_PARAMETERS:

    #Generate CNN Architecture

    parameter = math.floor(parameter)

    experiment = {}

    num_input_channels = 1
    num_output_channels = num_input_channels * parameter

    for layer in range(1, NUM_LAYERS + 1):
        next_layer = {layer : [num_input_channels, num_output_channels, KERNEL_SIZE, NUM_PADDING, STRIDE]}
        experiment.update(next_layer)

        num_input_channels = num_input_channels * parameter
        num_output_channels = num_input_channels * parameter

    print("Currently Testing with Parameter", parameter, " and experiment architecture ", experiment)
    model_name = EXPERIMENT_NAME + "_" + str(parameter)
    model_store.append(ModelGenerator(model_name))

    model_store[-1].generate_cnn_layers_model(experiment, INPUT_SIZE)
    model_store[-1].generate_fnn_layers_models(FFN_ARCHITECTURE)

print("-------------------- Generating Training Area --------------------")

for model in model_store:
    training_store.append(ModelTrainingFactory(model, LEARNING_RATE, EXPERIMENT_NAME))

    training_store[-1].train_model(train_loader, dev_test_loader, NUM_TRAINING_EPOCHS)

    training_store[-1].generate_summary_data()

    upload_to_bucket.upload_file(training_store[-1].get_summary_graph_name(), training_store[-1].get_summary_graph_name())


print("-------------------- Generating Overall Graph Factory Area --------------------")

graph_factory = GraphFactory(training_store)
graph_factory.set_graph_name(EXPERIMENT_NAME)
overall_graph_name = graph_factory.plot_graph()

upload_to_bucket.upload_file(overall_graph_name, overall_graph_name)