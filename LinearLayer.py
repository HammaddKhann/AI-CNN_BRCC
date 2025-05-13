import numpy as np


class LinearLayer:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.weights = np.random.rand(out_features, in_features).astype(np.float32)
        self.bias = np.random.rand(out_features).astype(np.float32)

        self.inputs = None
        self.has_weights = True

    def forward(self, inputs):
        # print("shape of inputs in linear layer forward method:  ", inputs.shape)
        self.inputs = inputs
        return np.dot(self.inputs, self.weights.T) + self.bias  # matrix multiplication between the input vector and
        # the weight matrix

    def backward(self, d_outputs):
        return_dict = {}
        # print("shape of weights in linear layer backward method:  ", self.weights.shape)
        # print("shape of weights (transposed) in linear layer backward method:  ", self.weights.T.shape)
        # print("shape of outputs in linear layer backward methodX:  ", d_outputs.shape)

        return_dict["d_weights"] = d_outputs.T @ self.inputs
        # computes the gradient of the loss with respect to the weights
        # matrix multiplication results in a matrix of shape (out_features, in_features)

        return_dict["d_bias"] = np.sum(d_outputs, axis=0)
        # Computes the sum of gradients along the batch dimension (axis=0). This results in a vector of shape
        #
        return_dict["d_out"] = d_outputs @ self.weights
        # results in a matrix of shape (batch_size, in_features)
        return return_dict

    def update(self, d_weights, d_bias, learning_rate):
        self.weights -= learning_rate * d_weights  # Update the weights of the fully connected layer
        self.bias -= learning_rate * d_bias  # Update the bias of the fully connected layer
        return
