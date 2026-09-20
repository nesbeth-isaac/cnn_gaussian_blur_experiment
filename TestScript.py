from DataLoaderFactory import produce_dataloader
from GausiannBlurGenerator import GausiannBlurGenerator
from ModelArchitecture import ModelGenerator
from TrainModel import ModelTrainingFactory
import matplotlib.pyplot as plt
import torch
gaussians = GausiannBlurGenerator(100, 91, -4, 4, -4, 4)
data, labels = gaussians.generate_gausiann_blurs()
train, dev, test = produce_dataloader(data, labels)

model = ModelGenerator("Test")

model.generate_cnn_layers_model({1 : [1, 10, 3, 1, 1], 2:  [10, 12, 3, 1, 1]}, 91)

model.generate_fnn_layers_models({1: [10, 1]})

training_area = ModelTrainingFactory(model, 0.1, "Experiment")

training_area.train_model(train, dev, 5)

training_area.generate_feature_map_summary_file()