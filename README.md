# Deep Dream Application

The Deep Dream App is a graphical user interface (GUI) application that allows you to apply Deep Dream algorithms to images using various parameters. Deep Dream is a computer vision technique that enhances images by altering them in a way that showcases the patterns and features that a neural network model has learned.

## Core Technologies Used

- **TensorFlow**: Deep Dream App utilizes TensorFlow, an open-source machine learning framework developed by Google, to load and manipulate neural network models such as Inception.
- **Inception Model**: The application uses the Inception model (e.g., Inception5h) for feature extraction and to perform Deep Dream operations on images.
- **PyQt5**: PyQt5 is used for creating the graphical user interface and interactive elements of the application.
- **PIL (Python Imaging Library)**: PIL is used to handle image loading, manipulation, and saving in the application.


## Features

- Load an image from your computer.
- Select a layer from the available layers to apply Deep Dream on.
- Adjust parameters such as channel, iterations, step size, num octaves, and octave scale.
- Apply Deep Dream to the loaded image with the selected parameters.
- Preview the deep dream image.
- Download the processed image to your computer.
- Exit the application.

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python 3.6+**: The application is built using Python programming language.
- **PyQt5**: You need to have PyQt5 installed to create the graphical user interface and interactive elements of the application.
- **TensorFlow (or compatible library)**: TensorFlow is used to load and manipulate neural network models like Inception.
- **PIL (Python Imaging Library)**: PIL is required to handle image loading, manipulation, and saving within the application.
- **NumPy**: NumPy is used for numerical operations and array manipulation in Python.
- **Matplotlib**: Matplotlib is used for creating visualizations, such as plotting images.
- **urllib**: The urllib library is used for opening URLs and downloading content from the internet.
- **os**: The os module provides functions to interact with the operating system, such as file operations.
- **zipfile**: The zipfile module provides tools to work with ZIP archives.

Ensure you have these libraries installed before running the application.

### Installation & Usage
1. Download the Zip 
    - **(step 1)** Find the green button "<> Code" and select "Download Zip"
    - **(step 2)** Unzip the folder 
    - **(step 3)** Run deep_dream_app.py 

2. Clone this repository:
    - **(step 1)** In your terminal/bash enter "git clone https://github.com/your-username/deep-dream-app.git" with your own username
    - **(step 2)** cd deep-dream-app
    - **(step 3)** Run deep_dream_app.py 


