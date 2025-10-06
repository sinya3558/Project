# TensorFlow ver.
# train_gqcnn_rgbd.py
import os
import json
# import sys
import numpy as np
from PIL import Image
import tensorflow as tf
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from gqcnn_repo.gqcnn.model.tf.network_tf import GQCNNTF
from tqdm import tqdm  # 진행률 표시

# load datasets
class CustomGraspDataset:
    def __init__(self, img_dir, ann_dir):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.files = []

        for f in os.listdir(img_dir):
            if not f.endswith(".jpg"):
                continue
            rgb_path = os.path.join(img_dir, f)
            depth_path = os.path.join(img_dir, f.replace(".jpg","_depth.png"))
            json_file = f.replace("TS_", "TL_").replace(".jpg",".json")
            json_path = os.path.join(ann_dir, json_file)

            if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(json_path):
                self.files.append(f)

        if len(self.files) == 0:
            raise ValueError("No data found. Check your data directory.")

    def __len__(self):
        return len(self.files)

    def get_batch(self, batch_size):
        for i in range(0, len(self.files), batch_size):
            batch_files = self.files[i:i+batch_size]
            imgs = []
            # label_text = []   # succ/fail 확인용
            label_int = []      # 0/1 트레이닝용
            for f in batch_files:
                # load rgb
                rgb = np.array(Image.open(os.path.join(self.img_dir, f)).resize((224,224)), dtype=np.float32)/255.0
                
                # load depth
                depth = np.array(Image.open(os.path.join(self.img_dir, f.replace(".jpg","_depth.png"))).resize((224,224)), dtype=np.float32)/255.0
                depth = np.expand_dims(depth, axis=-1)

                # rgb + depth -> 4 channel input
                img = np.concatenate([rgb, depth], axis=-1)
                imgs.append(img)

                # load label
                json_file = f.replace("TS_", "TL_").replace(".jpg",".json")
                with open(os.path.join(self.ann_dir, json_file), "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                    # grasp_result = int(data.get("grip_succeed", 0) )     # "grasp_succeed" exists in JSON files + 모델 학습용(1/0)
                    # label_text = "SUCCESS" if grasp_result == 1 else "FAIL"   # tags(text): "SUCCESS", "FAIL"

                    # label_text.append(label_text)     # error -> inference only
                    label_int.append(int(data.get("grip_succeed", 0)))

            yield np.array(imgs, dtype=np.float32), np.array(label_int, dtype=np.float32)


# GQ-CNN model network
##### => gqcnn_repo\gqcnn\model\tf\network_tf.py has issues w unsupported input data mode
def custom_gqcnn_rgbd(input_shape=(224,224,4)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # grasp success prob
    ])
    return model


# training function calls
if __name__ == "__main__":
    img_dir = "../data/imgs"
    ann_dir = "../data/annotations"
    save_path = "gqcnn_rgbd.h5"
    batch_size = 8
    epochs = 10
    lr = 1e-3

    # data
    dataset = CustomGraspDataset(img_dir, ann_dir)

    # model:custom gq-cnn
    model = custom_gqcnn_rgbd(input_shape=(224,224,4))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

    # train
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        batch_count = 0
        
        for batch_x, batch_y in tqdm(dataset.get_batch(batch_size), total=int(np.ceil(len(dataset)/batch_size)), desc="Training", ncols=80):
            history = model.train_on_batch(batch_x, batch_y)
            loss, acc = history
            total_loss += loss
            total_acc += acc
            batch_count += 1
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/batch_count:.4f} - Acc: {total_acc/batch_count:.4f}")

    # Save model
    model.save(save_path)
    print(f"Training done -> saved to {save_path}")
