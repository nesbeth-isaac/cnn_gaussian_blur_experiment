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

# Reading Config File
parametric_config = configparser.ConfigParser()
parametric_config.read('parametric_configuration.ini')

cloud_config = configparser.ConfigParser()
cloud_config.read('cloud_configuration.ini')
EXPERIMENT_NAME = str(parametric_config['parametric_experiment_1']['experiment_name'])
LINSPACE_STARTING_NODE = int(parametric_config['parametric_experiment_1']['fnn_architecture_np_linspace_start'])
LINSPACE_ENDING_NODE = int(parametric_config['parametric_experiment_1']['fnn_architecture_np_linspace_end'])
NUM_LAYERS = [int(num_node.strip()) for num_node in parametric_config['parametric_experiment_1']['fnn_architecture_np_linspace_num_layers'].split(',')]
CNN_ARCHITECTURE = {1 : [1, 10, 10, 1, 1], 2: [10, 50, 5, 1, 1]}
LEARNING_RATE = float(parametric_config['parametric_experiment_1']['learning_rate'])
NUM_TRAINING_EPOCHS = int(parametric_config['parametric_experiment_1']['num_training_epochs'])
INPUT_SIZE = int(parametric_config['common_attributes']['input_size'])

S3_BUCKET_ID = str(cloud_config['s3_bucket_information']['s3_bucket_id'])

upload_centre = UploadToBucket(S3_BUCKET_ID)

GaussianBlurSet = GausiannBlurGenerator(1000, 91, -4, 4, -4, 4)

image_data, label_data = GaussianBlurSet.generate_gausiann_blurs()

train_loader, dev_test_loader, final_test_loader = produce_dataloader(image_data, label_data)

#-----------------Starting Experiment------------------------------#

print("Starting Experiment - setting up experiment parameters")
print("Setting up Model Architecture")
models_store = []
training_hold = []

for current_num_layers in NUM_LAYERS:
    list_of_nodes = np.linspace(LINSPACE_STARTING_NODE, LINSPACE_ENDING_NODE, current_num_layers)

    fnn_architecture = {}

    current_layer = 1

    for i, node in enumerate(list_of_nodes):

        if i != len(list_of_nodes) - 1:
            fnn_architecture[current_layer] = [math.floor(node), math.floor(list_of_nodes[i + 1])]
            current_layer += 1

    print('Generated Architecture', fnn_architecture)

    #Generate Model

    model_name = "Model architecture with " + str((list(fnn_architecture.keys())[-1] + 1)) + " layers"


    models_store.append(ModelGenerator(model_name))

    models_store[-1].generate_cnn_layers_model(CNN_ARCHITECTURE, INPUT_SIZE)
    models_store[-1].generate_fnn_layers_models(fnn_architecture)

print("Set up Model Architectures")

print("Setting up Training Area")

for untrained_model in models_store:
    training_hold.append(ModelTrainingFactory(untrained_model, LEARNING_RATE, EXPERIMENT_NAME))

    print("Running Training On ", training_hold[-1].generated_model.model_name)

    #Train Model
    training_hold[-1].train_model(train_loader, dev_test_loader, NUM_TRAINING_EPOCHS)

print("Generating Summary Graphs for Each Graph")

for trained_model in training_hold:

    trained_model.generate_summary_data()

    upload_centre.upload_file(trained_model.get_summary_graph_name(),
                              trained_model.get_summary_graph_name())


print("Generating Summary Graph for All Experiments")

graph_factory = GraphFactory(training_hold)

graph_factory.set_graph_name(EXPERIMENT_NAME)
saved_file_name = graph_factory.plot_graph()

upload_centre.upload_file(saved_file_name, saved_file_name)

print("Completed Experiment")
