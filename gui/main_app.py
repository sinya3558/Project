from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QTextEdit, QVBoxLayout,QMessageBox,
    QWidget, QLabel, QHBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from api.openai_api import get_image_description
from utils.file_handler import get_image_file, encode_image_to_base64   # 안쓰이는뎅?
from utils.config import DB_PATH
import sqlite3
import os

# improt gqcnn model
from model.infer_gqcnn_rgbd import GQCNN

class MainWindow(QMainWindow):  # GUI CLASS
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manipulation Robot Grasp Quality Failure Reason Analysis")
        self.setGeometry(100, 100, 800, 600)
        self.image_path = None

        # init model to check grasp status
        self.gq_cnn = GQCNN(model_path = "model/gqcnn_rgbd_best.h5", classes_path="model/object_classes.json")   # 학습할 때 class 추출해둠

        self.init_ui()
        self.init_db()

    # Part 1 - UI
    def init_ui(self):
        self.image_label = QLabel("Upload your image")
        self.image_label.setFixedSize(300, 300)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")

        self.load_button = QPushButton("Open Image")
        self.load_button.clicked.connect(self.load_image)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("(Optional) Ask GPT why it didn't work")

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        self.result_output.setPlaceholderText("Result will be appeared here >>")

        self.generate_button = QPushButton("Process Image")
        # self.generate_button.clicked.connect(self.generate_description)
        self.generate_button.clicked.connect(self.on_process_image_clicked)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.image_label)
        top_layout.addWidget(self.load_button)

        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.text_input)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.result_output)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # Part 2 - DB # 여기가 이제 마지노선!
    def init_db(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # 쿼리 메세지 : success/fail, fail_reason(gpt), gpt_prompt
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT,
                image BLOB,
                grasp_status TEXT,
                probability REAL,
                fail_reason TEXT,
                prompt TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    # Part 3 - load img
    def load_image(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
            if path:
                pixmap = QPixmap(path).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
                if pixmap.isNull():
                    raise ValueError("Unable to load the image.")
                self.image_label.setPixmap(pixmap)
                self.image_path = path
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load the image: {e}")

    # Part 4 - manual GPT(prompt)
    def on_process_image_clicked(self):
        if not self.image_path:
            QMessageBox.information(self, "Notice", "Please select an image first.")
            return
        user_prompt = self.text_input.toPlainText().strip()
        self.process_image(user_prompt or None)

    # Part 5 - classify grasp + GPT fail reason
    def process_image(self, user_prompt= None):
        if not self.image_path:
            self.result_output.setPlainText("Please load an image first.")
            return
        
        try:
            # predict grasp status using gq-cnn (success/fail (+ probability %))
            predict_int, prob = self.gq_cnn.predict(self.image_path)  
            grasp_status = "SUCCESS" if predict_int else "FAIL"

            # if grasp == "FAIL", ask GPT
            fail_reason = None
            prompt = None
            if grasp_status == "FAIL":
                prompt = user_prompt or (
                    "List only the top 3 reasons why manipulation robot grasp failed in very short keywords separated by commas. Only give keywords, no full sentences.")
                fail_reason = get_image_description(self.image_path, prompt)

            # prompt = self.text_input.toPlainText()
            # base64_image = encode_image_to_base64(self.image_path)
            # result = get_image_description(self.image_path, prompt)

            # print out result on gui
            result = f"Grasp Result : {grasp_status}\nPercentage : {prob*100:.2f}%"   # probability * 100 => percentage
            if fail_reason: 
                result += f"\n\nGPT analysis on failure reasons : \n{fail_reason}"
            self.result_output.setPlainText(result)

            # save to DB
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                with open(self.image_path, "rb") as f:
                    image_blob = f.read()
                cursor.execute('''
                    INSERT INTO image_logs (image_path, image, grasp_status, probability, fail_reason, prompt)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (self.image_path, image_blob, grasp_status, prob, fail_reason, prompt))
                conn.commit()
        except Exception as e:
            QMessageBox.warning(self, "Processing Error", f"Failed to process image: {e}")
