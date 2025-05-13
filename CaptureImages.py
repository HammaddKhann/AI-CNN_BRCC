import cv2
import os
import random

# Function to capture images and save them with labels


def capture_images(output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize video capture
    camera = cv2.VideoCapture(0)

    # Counter for image naming
    count = 0

    # Capture loop
    while True:
        ret, frame = camera.read()
        cv2.imshow('Capture', frame)

        # Wait for key press (press 'c' to exit or 'c or v' to capture)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            label = 1     # 1 for extended arm
            image_path = os.path.join(output_dir, f"{label}_{count}.jpg")
            frame = cv2.resize(frame, (120, 120))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")
            count += 1
        elif key == ord('c'):
            label = 0     # 0 for contracted arm
            image_path = os.path.join(output_dir, f"{label}_{count}.jpg")
            frame = cv2.resize(frame, (120, 120))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")
            count += 1
        elif key == ord('x'):
            break

    # Release the capture
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Directory to store captured images
    output_dir_images = "arm_images"

    print("Capture images of extended arm (press 'e' to capture extended. 'c' for contracted, 'x' to quit)...")
    capture_images(output_dir_images)

