import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet")

# Function to classify an image
def classify_image():
    global panel
    file_path = filedialog.askopenfilename()
    
    if not file_path:
        return  # If no file is selected, exit function
    
    # Load and preprocess the image
    image = Image.open(file_path)
    image = image.resize((224, 224))  # Resize for MobileNetV2
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the class
    preds = model.predict(img_array)
    label = decode_predictions(preds, top=3)[0]  # Get top 3 predictions

    # Display image
    img_tk = ImageTk.PhotoImage(image)
    panel.config(image=img_tk)
    panel.image = img_tk

    # Display predictions
    result_text.set(f"Predictions:\n1. {label[0][1]} ({label[0][2]*100:.2f}%)\n"
                    f"2. {label[1][1]} ({label[1][2]*100:.2f}%)\n"
                    f"3. {label[2][1]} ({label[2][2]*100:.2f}%)")

# GUI setup
root = tk.Tk()
root.title("AI Image Classifier")
root.geometry("500x600")

# UI Elements
panel = tk.Label(root)
panel.pack(pady=10)

btn = tk.Button(root, text="Select Image", command=classify_image, font=("Arial", 14), bg="lightblue")
btn.pack(pady=10)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 12), fg="blue")
result_label.pack(pady=10)

# Run the GUI
root.mainloop()
