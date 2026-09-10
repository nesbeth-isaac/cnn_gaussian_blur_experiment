import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class ModelGenerator(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.network = nn.ModuleList()
        self.model_name = model_name

        self.num_pixels_per_layer = {}
        self.total_pixels_per_layer = {}

        self.latest_num_layers_cnn_layers = 0

    def calculate_num_pixels(self):
        pass


    def generate_cnn_layers_model(self, model_architecture, input_size):
        """model_architecture should be in the format:
            {layer1: [num_input_channels, num_output_channels, kernel_size, num_padding, num_stride]}

            input_size: represents the X x Y size of the image. For example, a 4 x 4 image would be represented as 4. """

        output_size_per_layer_1D = input_size
        latest_num_layers = 0
        for position, layer in enumerate(model_architecture.keys()):
            num_input_channels = model_architecture[layer][0]
            num_output_channels = model_architecture[layer][1]
            kernel_size = model_architecture[layer][2]
            num_padding = 0
            num_stride = 0
            if len(model_architecture[layer]) == 4:
                num_padding = model_architecture[layer][3]

            if len(model_architecture[layer]) == 5:
                num_padding = model_architecture[layer][3]
                num_stride = model_architecture[layer][4]


            self.network.append(nn.Conv2d(
                num_input_channels, num_output_channels, kernel_size,
                padding=num_padding, stride=num_stride
            ))

            output_size_per_layer_1D = self.calculate_output_pixels_1D(output_size_per_layer_1D, kernel_size, num_padding, num_stride)
            self.num_pixels_per_layer[position] = [output_size_per_layer_1D]
            self.total_pixels_per_layer[position] = [output_size_per_layer_1D * output_size_per_layer_1D * num_output_channels]
            latest_num_layers = num_output_channels

            self.network.append(nn.ReLU())
            self.network.append(nn.MaxPool2d(kernel_size=2, stride=2))

            output_size_per_layer_1D = self.calculate_output_pixels_1D(output_size_per_layer_1D, 2, 0, 2)
            self.num_pixels_per_layer[position].append(output_size_per_layer_1D)
            self.total_pixels_per_layer[position] = [output_size_per_layer_1D * output_size_per_layer_1D * num_output_channels]



        self.network.append(nn.Flatten())

        self.latest_num_layers_cnn_layers = latest_num_layers

    def generate_fnn_layers_models(self, model_architecture):
        """model_architecture should be in the format:
            {layer1: [num_input_features, num_output_features]}
             Note that the total number of pixels per layer should be calculated at the end of the CNN architecture.
             If this is the first layer of the FNN, the first element for inputs should be 0. """

        num_input_pixels = ((self.total_pixels_per_layer[list(self.total_pixels_per_layer.keys())[-1]])[-1])
        is_first_iteration = True

        for layer in model_architecture.keys():
            if is_first_iteration:
                self.network.append(nn.Linear(num_input_pixels, model_architecture[layer][1]))
                self.network.append(nn.ReLU())
                is_first_iteration = False

            else:
                self.network.append(nn.Linear(model_architecture[layer][0],
                                              model_architecture[layer][1]))
                self.network.append(nn.ReLU())



    def calculate_output_pixels_1D(self, input_size, kernel_size, padding, stride):
        height = math.floor((input_size + (2 * padding) - kernel_size) / stride) + 1

        return height

    def forward(self, data, is_training):

        current_value = data

        for layer in self.network:
            current_value = layer(current_value)

        if is_training == False:
            current_value = nn.functional.sigmoid(current_value)

        return current_value


