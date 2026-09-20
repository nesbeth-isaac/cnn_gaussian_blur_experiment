"""
File-name: TrainModel.py

Description: Class that houses ModelTrainingFactory.py
"""

import torch, torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt
import math

from ultralytics.trackers.utils import kalman_filter

from UploadToBucket import UploadToBucket
class ModelTrainingFactory:

    def __init__(self, generated_model, learning_rate, experiment_name):
        self.generated_model = generated_model

        self.learning_rate = learning_rate

        self.loss_function = BCEWithLogitsLoss()

        self.optimiser = torch.optim.SGD(self.generated_model.parameters(), lr=learning_rate)

        self.training_accuracy = None
        self.test_accuracy = None
        self.confusion_matrix = None
        self.training_loss = None

        self.confusion_matrix = None

        self.summary_graph_name = None

        self.summary_graph_name = None

        self.feature_map_name = None

        self.device = None

        self.__test_data = None

        self.__experiment_name = experiment_name

        self.feature_map_names = []

    def train_model(self, training_data, test_data, num_epochs):
        print("Training Model")
        self.training_loss = torch.zeros(num_epochs)
        self.training_accuracy = torch.zeros(num_epochs)
        self.test_accuracy = torch.zeros(num_epochs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generated_model.to(self.device)

        self.__test_data = next(iter(test_data))

        self.num_epochs = 0


        for epoch in range(0, num_epochs):
            self.num_epochs = num_epochs
            print("Training Epoch ", epoch, "out of ", num_epochs)

            num_batches = len(training_data)

            batch_count = 0

            batch_training_loss = torch.zeros(num_batches)
            batch_training_accuracy = torch.zeros(num_batches)

            for data, target_labels in training_data:

                data = data.to(self.device)
                target_labels = target_labels.to(self.device)

                prediction = self.generated_model.forward(data, is_training=True)

                loss = self.loss_function(prediction, target_labels)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                batch_training_loss[batch_count] = loss.detach().item()

                prediction = (prediction > 0.5).float()

                training_accuracy = prediction == target_labels

                training_accuracy = training_accuracy.float()

                batch_training_accuracy[batch_count] = torch.mean(training_accuracy) * 100

                batch_count = batch_count + 1

            self.training_loss[epoch] = torch.mean(batch_training_loss).item()
            self.training_accuracy[epoch] = torch.mean(batch_training_accuracy).item()

            self.run_test_set(test_data, epoch)

    def run_test_set(self, test_data, epoch):
        print("Testing Model")
        self.generated_model.eval()
        for data, labels in test_data:
            data = data.to(self.device)
            labels = labels.to(self.device)
            prediction = self.generated_model.forward(data, is_training=False)

            correctly_classified = (prediction > 0.5).float()

            correctly_classified = correctly_classified.float()

            correctly_classified = correctly_classified == labels

            correctly_classified = correctly_classified.float()

            self.test_accuracy[epoch] = correctly_classified.mean().item() * 100

            print(self.test_accuracy)

        self.generated_model.train()

    def generate_summary_data(self):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].plot(self.training_accuracy.detach())
        ax[0].set_title(f'Training Accuracy {self.generated_model.model_name}')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylim([0, 110])
        ax[0].set_xlim([0, self.num_epochs])

        ax[1].plot(self.training_loss.numpy())
        ax[1].set_title('Training Loss')
        ax[1].set_ylabel('Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylim([0, 1])
        ax[0].set_xlim([0, self.num_epochs])

        ax[2].plot(self.test_accuracy.detach())
        ax[2].set_title('Test Accuracy')
        ax[2].set_ylabel('Accuracy')
        ax[2].set_xlabel('Epoch')
        ax[2].set_ylim([0, 110])

        plt.plot()

        plt.savefig(self.summary_graph_name, format="png")

        print("File Saved")

    def set_summary_graph_name(self):
        self.summary_graph_name = self.__experiment_name + "_" + self.generated_model.model_name + "_Graph.png"

    def set_feature_map_name(self, layer_num, image_num):
        """
        Method name: set_feature_map_name()
        Description: Sets the name of a feature map and then adds the name to a list.
        :param layer_num:
        :return:
        """
        self.feature_map_name = (self.__experiment_name + "_" + self.generated_model.model_name + "_Feature_Map__Layer" + str(layer_num)
                                 + "_ImgNum_" + str(image_num) + ".png")
        self.feature_map_names.append(self.feature_map_name)

    def get_feature_map_names(self):
        return self.feature_map_names

    def print_latest_statistics(self):
        print("Last 3 Training Accuracy: ", self.training_accuracy[-1 : -3])
        print("Last 3 Loss: ", self.training_loss[-1:-3])
        print("Last 3 Test Accuracy", self.training_loss[-1:-3])

    def get_summary_graph_name(self):
        return self.summary_graph_name

    def get_feature_maps(self):
        """
        Method Name: get_feature_maps()
        :return: List of Tensors where each Tensor represents a layer containing a batch of X Feature Maps of size [X, Size, Size].
                 To access a specific feature map, you must index twice. First, to access Tensor (Layer) I and
                 then again to access the specific feature map J in that layer.
        """
        data, labels = self.__test_data
        output = self.generated_model.forward(data, is_training=False, save_feature_maps = True)


        return self.generated_model.get_feature_maps()

    def generate_feature_map_summary_file(self):
        """
        Method Name: generate_feature_map_summary_file()
        Method Description: method that generates a PNG for each layer, where each PNG
        contains all the feature maps in that layer.
        :return: None
        """
        feature_maps = self.get_feature_maps()


        for layer_num, layer in enumerate(feature_maps):

            layer = torch.squeeze(layer)
            layer_len = layer.shape[1]
            batch_len = layer.shape[0]

            segment = math.ceil(batch_len * 0.1)
            batch = layer[0:segment:,:,:]

            map_num = 0
            image_num = 1
            for image in batch:
                fig, ax = plt.subplots((math.ceil(layer_len / 3)), 3, figsize=(20, 10), squeeze=False)

                iterator = 0
                row = 0
                col = 0
                for feature_map in image:
                    self.set_feature_map_name(layer_num, image_num)
                    ax[row, col].imshow(feature_map, cmap="hot")
                    ax[row, col].axis('off')
                    map_name = f"Feature Map: {layer_num}_{map_num}"
                    ax[row, col].set_title(map_name)
                    iterator += 1
                    row = iterator // 3
                    col = iterator % 3
                    map_num += 1

                fig.suptitle(f"Feature Map Generation - {self.generated_model.model_name}")
                fig.savefig(self.feature_map_name, format="png")

                image_num += 1






















