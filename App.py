import pickle
import cv2
import Camera
import tkinter as tk
from tkinter import *
import numpy as np
import PIL.Image, PIL.ImageTk


class App:

    def __init__(self):

        self.window = tk.Tk()
        self.window.title = "AI PROJECT 2024"

        self.counting_enabled = False

        self.camera = Camera.Camera()

        self.canvas = None
        self.counter_label = None
        self.btn_reset = None
        self.btn_toggle = None

        self.extended = False
        self.contracted = False

        self.rep_counter = 0
        self.count = 0

        with open('CNN_MODEL_KERAS', "rb") as f:
            self.model = pickle.load(f)

        self.init_gui()

        self.delay = 200
        self.update()

        self.window.attributes("-topmost", True)
        self.window.mainloop()

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        self.btn_toggle = Button(self.window, text="Toggle Counting", command=self.counting_toggle)
        self.btn_toggle.pack(anchor=tk.CENTER, expand=True)

        self.counter_label = tk.Label(self.window, text=f"{self.rep_counter}")
        self.counter_label.config(font=("Arial", 24))
        self.counter_label.pack(anchor=tk.CENTER, expand=True)

    def update(self):
        if self.counting_enabled:
            self.prediction()

        if self.extended and self.contracted:
            print("Rep completed")
            self.extended, self.contracted = False, False
            self.rep_counter += 1

        ret, frame = self.camera.get_frame()
        if ret:
            # Convert the frame to RGB format for display in tkinter canvas
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize the frame to fit the canvas size
            frame_resized = cv2.resize(frame_rgb, (self.canvas.winfo_width(), self.canvas.winfo_height()))
            # Convert the frame to a PhotoImage object
            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_resized))
            # Update the canvas with the new frame
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            # Keep a reference to the PhotoImage object to prevent it from being garbage collected
            self.canvas.photo = photo

        self.counter_label.config(text=f"{self.rep_counter}")
        self.window.after(self.delay, self.update)

    def prediction(self):

        ret, frame = self.camera.get_frame()
        if not ret:  # If frame not captured successfully
            print("Error: Could not read frame.")
            return

        # print("Frame shape:", frame.shape)

        frame = cv2.resize(frame, (120, 120))  # Resize frame
        frame_rgb = frame[:, :, ::-1]  # Convert BGR to RGB

        # Preprocess input to match model's expectations
        frame_rgb = np.expand_dims(frame_rgb, axis=0)  # Add batch dimension
        frame_rgb = frame_rgb / 255.0  # Normalize pixel values

        prediction = self.model.predict(frame_rgb)
        print(prediction)

        prob_class_c = prediction[0][0]
        print(prob_class_c)
        prob_class_e = prediction[0][1]
        print(prob_class_e)
        if self.count == 0:
            if prob_class_e >= 0.88:
                print("The arm is extended")
                self.extended = True
                self.count += 1
        elif self.count == 1:
            if 0.80 <= prob_class_e < 0.895:
                print("The arm is contracted")
                self.contracted = True
                self.count = 0

    def counting_toggle(self):
        self.counting_enabled = not self.counting_enabled

    def reset(self):
        self.rep_counter = 0


def main():
    App()


if __name__ == "__main__":
    main()
