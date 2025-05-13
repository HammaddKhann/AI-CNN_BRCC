import numpy as np


class ReLU:  # to introduce non-linearity to the network, which allows the model to learn complex patterns in the data
    # Allows only some neurons to be active and not all of them
    def __init__(self):
        self.inputs = None
        self.has_weights = False

    def forward(self, inputs):
        # inputs -> The array from convolution layer forward method
        # outputs -> array of the same shape as the input but now the pixel values are either positive or 0
        self.inputs = inputs
        return np.maximum(0, inputs)  # Relu Function -> f(x)=max(0,x)

    def backward(self, d_outputs):
        return {"d_out": d_outputs * (self.inputs > 0)}
