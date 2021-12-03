from functools import partial

from PyQt5.QtWidgets import (QGroupBox, QHBoxLayout, QPushButton, QVBoxLayout,
                             QWidget)

from .train import Train


class Window(QWidget):
    def __init__(self):
        super().__init__()

        # layout setting
        self.padding = 20
        self.row_amount = 4
        self.col_amount = 4
        self._height = 400
        self._width = 500

        # window setting
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self.setWindowTitle("HW1-5")
        self.setGeometry(0, 0, self._width, self._height)
        self.init_btn()
        self.show()

    def init_btn(self):
        msgs = [
            "1. Show Train Images",
            "2. Show HyperParameter",
            "3. Show Model Struct",
            "4. Show Accuracy",
        ]
        v_layout = QVBoxLayout()
        group_box = QGroupBox("VGG16 TEST")
        for i in range(len(msgs)):
            btn = QPushButton(msgs[i], self)
            btn.clicked.connect(partial(self._executor, i))
            v_layout.addWidget(btn)
        group_box.setLayout(v_layout)
        self.layout.addWidget(group_box)

    def _executor(self, i):
        _train = Train(self)
        func = [
            getattr(_train, "show_img"),
            getattr(_train, "show_hyperparameters"),
            getattr(_train, "show_model"),
        ]
        if i < len(func):
            func[i]()
