import numpy as np


class Flatten:
    def __init__(self):
        self.inputs_shape = None
        self.has_weights = False

    def forward(self, inputs):
        # print("shape of inputs in flattening layer forward method:  ", inputs.shape)
        # The above printing statement is used to understand the number of inputs in the fully connected layer

        self.inputs_shape = inputs.shape  # for reshaping the input back to its original shape during the backward pass
        return inputs.reshape(inputs.shape[0], -1)  # Reshape the input into a single vector

    def backward(self, d_outputs):
        # print("shape of outputs in flattening layer backward method:  ", d_outputs.shape)
        # print("Shape of d_outputs before reshape:", d_outputs.shape)
        return {"d_out": d_outputs.reshape(self.inputs_shape)}
