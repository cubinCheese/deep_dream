# frontend of deep dream application
# responsible for setting UI functionality

import sys
import os
import numpy as np
import PIL.Image
import tensorflow as tf
from functools import partial
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QSlider
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from deep_dream import graph, deepdream

# Define a dark color scheme
dark_color_scheme = {
    'background_color': 'rgb(30, 30, 30)',
    'text_color': 'white',
    'button_background_color': 'rgb(50, 50, 50)',
    'button_text_color': 'white',
    'slider_background_color': 'rgb(50, 50, 50)',
    'slider_handle_color': 'rgb(0, 122, 204)',  # A blue color for the slider handle
}

# valid layer checking -- not strictly necessary -- currently only applying one combination layer option
def isLayerValid(layer):
    try:
        img0 = PIL.Image.new('RGB', (1, 1))
        img0 = np.float32(img0)
        channel = 0
        iterations = 1  
        step_size = 1.0
        num_octaves = 1
        octave_scale = 1.0
        t_obj = graph.get_tensor_by_name(f'import/{layer}:0')
        img_result = deepdream(t_obj[:, :, :, channel], img0, iter_n=iterations, step=step_size, octave_n=num_octaves, octave_scale=octave_scale)
        return True
    except:
        return False

class DeepDreamApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Deep Dream App')
        self.setGeometry(100, 100, 800, 600)

        # Set the application style to Fusion (for dark mode)
        QApplication.setStyle('Fusion')

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.load_button = QPushButton('Load Image')
        self.load_button.clicked.connect(self.loadImage)
        self.layout.addWidget(self.load_button)

        self.layers_label = QLabel('Select Layer:')
        self.layout.addWidget(self.layers_label)

        # List of layers (you can customize this)
        self.layers = ['mixed4d_3x3_bottleneck_pre_relu'] # not working: mixed4c_pool_reduce_pre_relu, 'mixed3c_pool_reduce_pre_relu'

        self.layer_buttons = []
        for layer in self.layers:
            if isLayerValid(layer):
                button = QPushButton('Apply Layer')
                button.clicked.connect(partial(self.applyDeepDream, layer))
                self.layer_buttons.append(button)
                self.layout.addWidget(button)
        
        # Creating 'Channel' slider 
        self.channel_label = QLabel('Channel:')
        self.channel_label.setStyleSheet(f'color: {dark_color_scheme["text_color"]};')  # Set text color to white
        self.channel_slider = QSlider(Qt.Horizontal)
        self.channel_slider.setMinimum(0)
        self.channel_slider.setMaximum(255)
        self.layout.addWidget(self.channel_label)
        self.layout.addWidget(self.channel_slider)
        self.channel_slider.valueChanged.connect(self.updateChannelValue) # displays current value of slider

        # Creating 'Iterations' slider
        self.iterations_label = QLabel('Iterations:')
        self.iterations_label.setStyleSheet(f'color: {dark_color_scheme["text_color"]};')  # Set text color to white
        self.iterations_slider = QSlider(Qt.Horizontal)
        self.iterations_slider.setMinimum(1)
        self.iterations_slider.setMaximum(100)
        self.layout.addWidget(self.iterations_label)
        self.layout.addWidget(self.iterations_slider)
        self.iterations_slider.valueChanged.connect(self.updateIterationsValue) # displays current value of iterations

        # Creating 'step size' slider
        self.step_size_label = QLabel('Step Size:')
        self.step_size_label.setStyleSheet(f'color: {dark_color_scheme["text_color"]};')  # Set text color to white
        self.step_size_slider = QSlider(Qt.Horizontal)
        self.step_size_slider.setMinimum(1)
        self.step_size_slider.setMaximum(100)
        self.layout.addWidget(self.step_size_label)
        self.layout.addWidget(self.step_size_slider)
        self.step_size_slider.valueChanged.connect(self.updateStepSizeValue) # displays current value of step size

        # Creating 'number of octaves' slider
        self.num_octaves_label = QLabel('Num Octaves:')
        self.num_octaves_label.setStyleSheet(f'color: {dark_color_scheme["text_color"]};')  # Set text color to white
        self.num_octaves_slider = QSlider(Qt.Horizontal)
        self.num_octaves_slider.setMinimum(1)
        self.num_octaves_slider.setMaximum(10)
        self.layout.addWidget(self.num_octaves_label)
        self.layout.addWidget(self.num_octaves_slider)
        self.num_octaves_slider.valueChanged.connect(self.updateNumOctavesValue) # displays current value of number of octaves

        # Creating 'octave scale' slider
        self.octave_scale_label = QLabel('Octave Scale:')
        self.octave_scale_label.setStyleSheet(f'color: {dark_color_scheme["text_color"]};')  # Set text color to white
        self.octave_scale_slider = QSlider(Qt.Horizontal)
        self.octave_scale_slider.setMinimum(1)
        self.octave_scale_slider.setMaximum(100)
        self.layout.addWidget(self.octave_scale_label)
        self.layout.addWidget(self.octave_scale_slider)
        self.octave_scale_slider.valueChanged.connect(self.updateOctaveScaleValue) # displays current value of octave scale

        # Creates download button for user to save output image
        self.download_button = QPushButton('Download')
        self.download_button.setStyleSheet(f'color: {dark_color_scheme["text_color"]};')  # Set text color to white
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.downloadImage)
        self.layout.addWidget(self.download_button)

        # Creates exit button within UI
        self.exit_button = QPushButton('Exit Application')
        self.exit_button.setStyleSheet(f'color: {dark_color_scheme["text_color"]};')  # Set text color to white
        self.exit_button.clicked.connect(self.exitApplication)
        self.layout.addWidget(self.exit_button)

        # Set styles for dark mode
        self.central_widget.setStyleSheet(f'background-color: {dark_color_scheme["background_color"]};')
        self.layers_label.setStyleSheet(f'color: {dark_color_scheme["text_color"]};')
        self.load_button.setStyleSheet(f'background-color: {dark_color_scheme["button_background_color"]}; color: {dark_color_scheme["button_text_color"]};')
        
        for button in self.layer_buttons:
            button.setStyleSheet(f'background-color: {dark_color_scheme["button_background_color"]}; color: {dark_color_scheme["button_text_color"]};')

        self.channel_label.setStyleSheet(f'color: {dark_color_scheme["text_color"]};') # Set styles for other labels, buttons, and sliders
        
        self.central_widget.setLayout(self.layout)

        self.input_image_path = ''
        self.output_image_path = ''

    # series of functions that update the UI with slider values as it is changed
    def updateChannelValue(self, value):
        self.channel_label.setText(f'Channel: {value}')

    def updateIterationsValue(self, value):
        self.iterations_label.setText(f'Iterations: {value}')

    def updateStepSizeValue(self, value):
        self.step_size_label.setText(f'Step Size: {value / 10.0:.1f}')

    def updateNumOctavesValue(self, value):
        self.num_octaves_label.setText(f'Num Octaves: {value}')

    def updateOctaveScaleValue(self, value):
        self.octave_scale_label.setText(f'Octave Scale: {value / 10.0:.1f}')

    # User specifies image to input for deep dream algorithm
    def loadImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            self.input_image_path = file_name
            pixmap = QPixmap(self.input_image_path)
            self.image_label.setPixmap(pixmap)

    # Utilizes user defined parameters, calling backend function of deep dream implementation
    def applyDeepDream(self, layer):
        if self.input_image_path:
            img0 = PIL.Image.open(self.input_image_path)
            img0 = np.float32(img0)

            # Get user-selected values from the sliders
            channel = self.channel_slider.value()
            iterations = self.iterations_slider.value()
            step_size = self.step_size_slider.value() / 10.0
            num_octaves = self.num_octaves_slider.value()
            octave_scale = self.octave_scale_slider.value() / 10.0

            # Check if the sliders have not been modified (using default values)
            if channel == 0 and iterations == 1 and step_size == 0.1 and num_octaves == 1 and octave_scale == 0.1:
                # Use default values for deep dream parameters
                channel = 139
                iterations = 20
                step_size = 1.5
                num_octaves = 4
                octave_scale = 1.4
            
            # try-catch to prevent application from crashing on unexpected errors
            try:
                # Apply Deep Dream
                t_obj = graph.get_tensor_by_name(f'import/{layer}:0')
                img_result = deepdream(t_obj[:, :, :, channel], img0, iter_n=iterations, step=step_size, octave_n=num_octaves, octave_scale=octave_scale)

                # Save the output image temporarily
                self.output_image_path = 'deep_dream_output.jpg'
                img_result = np.clip(img_result, 0, 255).astype(np.uint8)
                PIL.Image.fromarray(img_result).save(self.output_image_path)

                # Enable the download button
                self.download_button.setEnabled(True)

                # Preview the output image
                pixmap = QPixmap(self.output_image_path)
                self.image_label.setPixmap(pixmap)

                self.clearErrorMessage()
            # the UI application will instead indicate that parameter inputs lead to no result "nothing happened..."
            except Exception as e:
                print(f"An error occurred: {e}")
                self.showErrorMessage("Nothing happened...")

    # Print error message to indicate Nothing happened
    def showErrorMessage(self, message):
        #error_label = QLabel(message)
        #self.layout.addWidget(error_label)
        
        # creates widget to display text warning "nothing happened", and sets the text on concurrent warnings.
        if hasattr(self, 'error_label') and isinstance(self.error_label, QLabel):
            self.error_label.setText(message)
            self.error_label.show()
        else:
            self.error_label = QLabel(message)
            self.layout.addWidget(self.error_label)

    # if "Apply Layer" button is pressed, but result is achieved: clear failed warning text
    def clearErrorMessage(self):
        if hasattr(self, 'error_label') and isinstance(self.error_label, QLabel):
            self.error_label.clear()
            self.error_label.hide()

    # User may select location to download deep dream image
    def downloadImage(self):
        if self.output_image_path:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.jpeg);;All Files (*)")
            if save_path:
                os.rename(self.output_image_path, save_path)

    # functionality to exit application
    def exitApplication(self):
        QApplication.quit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DeepDreamApp()
    window.show()

    # remove temporary deep dream image
    if os.path.exists('temp_deep_dream.jpg'):
        os.remove('temp_deep_dream.jpg')

    sys.exit(app.exec_())
