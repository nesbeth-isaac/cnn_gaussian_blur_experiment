import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import torch.nn as nn
import random

""" Class Name: GaussianBlurGenerator()
    Purpose: Class that generates a set of Gausiann Blurs"""

class GausiannBlurGenerator():

    """
    Initialising Method.

    Input:
        - num_gausiaan: int Number of gaussians you wish to generate.
        - image_size = int.  Size of width and height of image. For example, when set to 20 then a 20 x 20 image will be generated.
        - start_x = int.     Lower bound for random generation of X coordinates.
        - end_x = int.       Upper bound of random generation for X coordinates.
        - start_y = int.     Lower bound of random generation for Y coordinates.
        - end_y = int.       Upper bound of random generation for Y coordinates.
    Output:
        - No output
    """
    def __init__(self, num_gaussian, image_size, start_x, end_x, start_y, end_y):
        self.num_gaussian = num_gaussian
        self.image_size = image_size

        self.initalised_x_positions = np.linspace(start_x, end_x, image_size)
        self.initalised_y_positions = np.linspace(start_y, end_y, image_size)

        self.x_positions, self.y_positions = np.meshgrid(self.initalised_x_positions, self.initalised_y_positions)

        self.widths_label_0 = np.linspace(1, 3, int(self.num_gaussian / 2))

        self.widths_label_1 = np.linspace(3.5, 5.5, int(self.num_gaussian / 2))

        self.images = torch.zeros(num_gaussian, 1, image_size, image_size)

        self.labels = torch.zeros(num_gaussian)

    """ 
        Method that generates Gaussian Blurs based on the parameters set in the __init__ method. 
    
        Input:
            - No Input
        
        Output:
            - self.images: 4D Tensor (a, b, c, d). Where a refers to the image number, b represents 
        """
    def generate_gausiann_blurs(self):

        for i in range(0, self.num_gaussian):
            random_centre_offset = 1 * random.randint(0, 2)

            width, self.labels[i] = self.get_random_width(i)

            gaussian = np.exp((-((self.x_positions - random_centre_offset) ** 2 +
                                 (self.y_positions - random_centre_offset) ** 2))
                              / 2 * (width**2))

            gaussian = gaussian + np.random.randn(self.image_size, self.image_size) / 5

            self.images[i, :, :, :] = torch.tensor(gaussian).view(1, self.image_size, self.image_size)

        return self.images, self.labels.reshape(-1, 1)

    def get_random_width(self, index_value):
        random_generator = random.randint(0,1)

        if random_generator == 1:
            return self.widths_label_1[math.floor(index_value / 2)], 1

        else:
            return self.widths_label_0[math.floor(index_value / 2)], 0




