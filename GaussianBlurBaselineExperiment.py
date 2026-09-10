import matplotlib.pyplot as plt
import numpy as np

from UploadToBucket import UploadToBucket
from GausiannBlurGenerator import GausiannBlurGenerator
from DataLoaderFactory import *
from ModelArchitecture import ModelGenerator
from TrainModel import ModelTrainingFactory

GaussianBlurSet = GausiannBlurGenerator(1000, 91, -4, 4, -4, 4)

image_data, label_data = GaussianBlurSet.generate_gausiann_blurs()

fig,axs = plt.subplots(3,7,figsize=(10,5))


for i,ax in enumerate(axs.flatten()):
  whichpic = np.random.randint(100)
  G = np.squeeze( image_data[whichpic,:,:] )
  ax.imshow(G, cmap = "hot")
  ax.set_xticks([])
  ax.set_yticks([])

plt.show()

x = np.linspace(10,20,5)
y = np.linspace(7, 13, 4)

train_loader, dev_test_loader, final_test_loader = produce_dataloader(image_data, label_data)

model = ModelGenerator("Baseline_Experiment")

cnn_architecture = {1 : [1, 10, 10, 1, 1], 2: [10, 50, 5, 1, 1]}

fnn_architecture = {1: [0, 1]}

model.generate_cnn_layers_model(cnn_architecture, 91)
model.generate_fnn_layers_models(fnn_architecture)

model_trainer = ModelTrainingFactory(model, 0.01)

model_trainer.train_model(train_loader, dev_test_loader, 5)

# model_trainer.generate_summary_data()
#
# upload = UploadToBucket("sagemaker-us-east-1-504895205985")
# upload.upload_file(f'{model_trainer.generated_model.model_name} Graph.png',
#               f'{model_trainer.generated_model.model_name} Graph.png')
#
# print("Completed")