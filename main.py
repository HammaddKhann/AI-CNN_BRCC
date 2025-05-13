from ConvolutionLayer import ConvolutionLayer
from LinearLayer import LinearLayer
from Flatten import Flatten
from ActivationFunction import ReLU
from LossFunction import CrossEntropy
from model import Model
from utils import load_dataset, compute_accuracy
import pickle

EPOCHS = 35
LEARNING_RATE = 0.01
BATCH_SIZE = 100   # Training model on 3500 images


layer_list = [
    ConvolutionLayer(in_channels=1, out_channels=10, kernel_size=5),
    ReLU(),
    Flatten(),
    LinearLayer(134560, 2),   # 116 x 116 (Output of convolution layer) x 10(Number of output channels) = 134560 (Num of Neurons after flattening)
    ReLU(),
]


def main():
    """
    x_train, x_test, y_train, y_test = load_dataset()
    print("Shape of x_train before flattening:", x_train.shape)
    print("Shape of x_test before flattening:", x_test.shape)

    Done to calculate the number of inputs for fully connected layer correctly
    """

    x_train, x_test, y_train, y_test = load_dataset()

    model = Model(layer_list, CrossEntropy())  # Assign layers and loss function to model object

    print("-" * 80)
    print(
        f"\tTraining Model: Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}, Learning Rate: {LEARNING_RATE}"
    )
    print("-" * 80)

    model.train(x_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)
    print("... Training Complete")

    with open('CNN_MODEL_SCRATCH', 'wb') as f:
        pickle.dump(model, f)

    predictions = model.predict(x_test)   # Predicted here to calculate the accuracy of the model
    predictions = predictions.argmax(axis=1)
    accuracy = compute_accuracy(predictions, y_test)   # Accuracy of model -> 60.17%
    print("==============================")
    print(f"\tAccuracy of your CNN model: {accuracy:.2%}")
    print("==============================")



if __name__ == "__main__":
    main()
