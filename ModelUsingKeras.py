import keras
import numpy as np
import random
from utils import load_dataset
import matplotlib.pyplot as plt
import pickle
from keras.api.models import Sequential
from keras.api.layers import Dense, Conv2D, MaxPooling2D, Flatten


def main():
    x_train, x_test, y_train, y_test = load_dataset()

    x_train = x_train.reshape(-1, 120, 120, 3)  # Reshaping to pass the correct shape in conv2d
    x_test = x_test.reshape(-1, 120, 120, 3)

    model = Sequential([
        Conv2D(300, (5, 5), activation='relu', input_shape=(120, 120, 3)),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(300, activation='relu'),
        Dense(2, activation='sigmoid')
    ])
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, batch_size=360)   #training on 3600 images

    with open('CNN_MODEL_KERAS', 'wb') as f:
        pickle.dump(model, f)

    print("Accuracy: ", model.evaluate(x_test, y_test))

    index = random.randint(0, len(x_test))
    image_data = x_test[index]
    plt.imshow(image_data)
    plt.show()

    y_predict = model.predict(x_test[index]).reshape(1, 120, 120, 3)
    print(y_predict)

if __name__ == "__main__":
    main()
