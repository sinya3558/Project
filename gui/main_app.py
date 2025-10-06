from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QTextEdit, QVBoxLayout,QMessageBox,
    QWidget, QLabel, QHBoxLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from api.openai_api import get_image_description
from utils.file_handler import get_image_file, encode_image_to_base64   # 안쓰이는뎅?
from utils.config import DB_PATH
import sqlite3

# improt gqcnn model
from model.gq_cnn import GQCNN

class MainWindow(QMainWindow):  # GUI CLASS
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manipulation Robot Grasp Predictor")
        self.setGeometry(100, 100, 800, 600)
        self.image_path = None

        # init model to check grasp status
        self.gq_cnn = GQCNN()

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

        self.generate_button = QPushButton("Process Image (Grasp + GPT)")
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

    # Part 2 - DB
    def init_db(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # 쿼리 메세지 : success/fail, fail_reason(gpt), gpt_prompt
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image BLOB,
                grasp_status TEXT,
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
            path = get_image_file()
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
        manual_text = self.text_input.toPlainText()
        self.process_image(manual_prompt = manual_text if manual_text else None)

    # Part 5 - classify grasp + GPT fail reason
    def process_image(self, manual_prompt: str = None):
        if not self.image_path:
            self.result_output.setPlainText("Please load an image first.")
            return
        
        try:
            # predict grasp status using gq-cnn (success/fail (+ probability %))
            prediction = self.gq_cnn.predict(self.image_path)
            grasp_status = "SUCCESS" if prediction else "FAIL"

            # if grasp == "FAIL", ask GPT
            fail_reason = None
            prompt = None
            if grasp_status == "FAIL":
                prompt = manual_prompt or (
                    "List only the top 3 reasons why manipulation robot grasp failed in very short keywords separated by commas. Only give keywords, no full sentences.")
                fail_reason = get_image_description(self.image_path, prompt)

            # prompt = self.text_input.toPlainText()
            # base64_image = encode_image_to_base64(self.image_path)
            # result = get_image_description(self.image_path, prompt)

            # print out result
            result = f"Grasp: {grasp_status}"   # ({percentage*100:.1f}%)
            if fail_reason:
                result += f"\nFail Reason (Automated by GPT): {fail_reason}"
            self.result_output.setPlainText(result)

            # save to DB
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                with open(self.image_path, "rb") as f:
                    image_blob = f.read()
                cursor.execute('''
                    INSERT INTO image_logs (image, grasp_status, fail_reason, prompt)
                    VALUES (?, ?, ?, ?)
                ''', (image_blob, grasp_status, fail_reason, prompt))
                conn.commit()
        except Exception as e:
            QMessageBox.warning(self, "Processing Error", f"Failed to process image: {e}")
