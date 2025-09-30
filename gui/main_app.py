from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QTextEdit, QVBoxLayout,QMessageBox,
    QWidget, QLabel, QHBoxLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from api.openai_api import get_image_description
from utils.file_handler import get_image_file, encode_image_to_base64
from utils.config import DB_PATH
import sqlite3

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenAI 이미지 설명 프로그램")
        self.setGeometry(100, 100, 700, 500)
        self.image_path = None
        self.init_ui()
        self.init_db()

    def init_ui(self):
        self.image_label = QLabel("이미지를 불러오세요")
        self.image_label.setFixedSize(300, 300)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")

        self.load_button = QPushButton("이미지 열기")
        self.load_button.clicked.connect(self.load_image)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("GPT에게 보낼 추가 프롬프트 입력")

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)

        self.generate_button = QPushButton("GPT 설명 생성")
        self.generate_button.clicked.connect(self.generate_description)

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

    def init_db(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # 쿼리 메세지
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image BLOB,
                prompt TEXT,
                response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def load_image(self):
        try:
            path = get_image_file()
            if path:
                pixmap = QPixmap(path).scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
                if pixmap.isNull():
                    raise ValueError("이미지를 불러올 수 없습니다.")
                self.image_label.setPixmap(pixmap)
                self.image_path = path
        except Exception as e:
            QMessageBox.warning(self, "오류", f"이미지 불러오기 실패: {e}")


# gen_description : gpt 설명 생성 부분 -> 이걸 이제 db에 저장할 수 잇게 해보자!
    def generate_description(self):
        if not self.image_path:
            self.result_output.setPlainText("이미지를 먼저 불러와 주세요.")
            return
        
        prompt = self.text_input.toPlainText()


        base64_image = encode_image_to_base64(self.image_path)
        result = get_image_description(self.image_path, prompt)
        self.result_output.setPlainText(result)

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            with open(self.image_path, "rb") as f:
                image_blob = f.read()
            cursor.execute('''
                INSERT INTO image_logs (image, prompt, response) VALUES (?, ?, ?)
            ''', (image_blob, prompt, result))
            conn.commit()

