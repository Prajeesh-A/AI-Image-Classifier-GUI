# AI Image Classifier GUI

A simple Tkinter-based GUI application that classifies objects in images using the MobileNetV2 deep learning model.

## Features
- Loads images using a file dialog.
- Uses MobileNetV2 for object classification.
- Displays the top 3 predicted labels with confidence scores.

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the application:
   ```sh
   python app.py
   ```

## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- Pillow
- Tkinter (built-in in Python)

## Example Usage
1. Click the "Select Image" button.
2. Choose an image file (JPG, PNG, etc.).
3. The model predicts and displays the top 3 object classes.

## License
This project is licensed under the MIT License.
