from PyQt5.QtWidgets import QFileDialog
import base64

def get_image_file():
    path, _ = QFileDialog.getOpenFileName(None, '이미지 선택', '', 'Images (*.png *.jpg *.jpeg)')
    return path


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

