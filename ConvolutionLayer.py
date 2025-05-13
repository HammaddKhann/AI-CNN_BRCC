import numpy as np


class ConvolutionLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.weights = np.random.rand(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(np.float32)
        self.bias = np.random.rand(out_channels).astype(
            np.float32)  # bias term allows the model to learn an offset that shifts the activation function and
        # helps in fitting the data better.

        self.inputs = None
        self.has_weights = True

    def forward(self, inputs):
        # input -> (batch_size, in_channels, height, width)
        # output -> (batch_size, out_channels, new_height, new_width)

        self.inputs = inputs
        batch_size = inputs.shape[0]
        in_height = inputs.shape[2]
        in_width = inputs.shape[3]
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1
        return_array = np.zeros((batch_size, self.out_channels, out_height, out_width))  # To store the output
        # feature maps

        for current_batch in range(batch_size):
            for current_out_channel in range(self.out_channels):  # For each example and output channel, calculate
                # the output feature map
                for current_height in range(out_height):
                    for current_width in range(out_width):
                        pre_height = current_height * self.stride  # starting height index of the current slice
                        post_height = pre_height + self.kernel_size  # the ending height index of the current slice
                        pre_width = current_width * self.stride  # starting width index of the current slice.
                        post_width = pre_width + self.kernel_size  # ending width index of the current slice.
                        input_slice = inputs[current_batch, :, pre_height:post_height, pre_width:post_width]  #
                        # extracts a slice from the input
                        # Colon is used to select all channels in the input.
                        return_array[current_batch, current_out_channel, current_height, current_width] = np.sum(
                            input_slice * self.weights[current_out_channel]) + self.bias[current_out_channel]  #
                        # compute the dot product between the input slice and the filter.

        # print("shape of inputs in convolution layer forward method:  ", inputs.shape)
        # print("Shape of return array from convolution layer forward pass: ", return_array.shape)
        return return_array

    def backward(self, d_outputs):

        return_dict = {}
        return_dict["d_weights"] = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)) # # Initializes an array of zeros to store the gradients of the loss with respect to the weights.
        return_dict["d_bias"] = np.zeros(self.bias.shape)  # Initializes an array of zeros to store the gradients of the loss with respect to the biases.
        return_dict["d_out"] = np.zeros(self.inputs.shape) # Initializes an array of zeros to store the gradients of the loss with respect to the inputs.

        batch_size = d_outputs.shape[0]
        out_height = d_outputs.shape[2]
        out_width = d_outputs.shape[3]

        for current_batch in range(batch_size):
            for current_out_channel in range(self.out_channels):
                for current_in_channel in range(self.in_channels):
                    for current_height in range(out_height):
                        for current_width in range(out_width):
                            pre_height = current_height * self.stride
                            post_height = pre_height + self.kernel_size
                            pre_width = current_width * self.stride
                            post_width = pre_width + self.kernel_size

                            #gradients are computed using the chain rule of calculus and involve multiplications between the input slices, weight slices, and the output gradients.

                            input_slice = self.inputs[current_batch, current_in_channel, pre_height:post_height,pre_width:post_width]

                            return_dict["d_weights"][current_out_channel, current_in_channel] += input_slice * d_outputs[current_batch, current_out_channel, current_height, current_width]
                            return_dict["d_bias"][current_out_channel] += d_outputs[current_batch, current_out_channel, current_height, current_width]
                            return_dict["d_out"][current_batch, current_in_channel, pre_height:post_height,pre_width:post_width] += self.weights[current_out_channel, current_in_channel] * d_outputs[
                                    current_batch, current_out_channel, current_height, current_width]

        # print("shape of outputs in convolution layer backward method:  ", d_outputs.shape)
        return return_dict

    def update(self, d_weights, d_bias, learning_rate):

        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias
        return
