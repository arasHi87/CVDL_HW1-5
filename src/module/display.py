import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QMainWindow, QSlider, QVBoxLayout, QWidget


class DisplayWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.slider = QSlider(Qt.Horizontal)
        self.public_frame = QLabel()
        self.widget = QWidget()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.public_frame)
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

    def convert_img(self, img):
        bytes_per_line = img.shape[1] * 3
        q_image = QImage(
            img.data, img.shape[1], img.shape[0], bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        return QPixmap.fromImage(q_image)

    def add_img_to_window(self, img):
        # add new label to insert image
        img_frame = QLabel()
        self.layout.addWidget(img_frame)
        img_frame.setPixmap(self.convert_img(img))

    def add_img_to_window_with_slider(self, img1, img2):
        self.slider.setRange(0, 255)
        self.slider.valueChanged.connect(lambda: self.slider_value_change(img1, img2))
        self.layout.addWidget(self.slider)
        self.slider_value_change(img1, img2)

    def slider_value_change(self, img1, img2):
        value = self.slider.value() / 255
        result = cv2.addWeighted(img1, 1 - value, img2, value, 0)
        self.public_frame.setPixmap(self.convert_img(result))