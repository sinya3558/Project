import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

#아.... pretained model 이 없다......
class GQCNN:
    def __init__(self, model_path="model/gqcnn_rgbd_tf"):
        if not os.path.exists(model_path):  
            raise FileNotFoundError(f"Trained model not found at {model_path}. Please train it first.")
        self.model = load_model(model_path) # load weight from trained model
        # print(f"load trained gq cnn model success!")

    def predict(self, image_path):
        # load imgs
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image data not found: {image_path}")
        img = Image.open(image_path).convert("RGB")

        # load depth
        depth_path = image_path.replace(".jpg","_depth.png")
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth data not found: {depth_path}")
        depth = Image.open(depth_path)
        
        # preprocess data (img+depth)
        rgb = np.array(img.resize((224,224)), dtype=np.float32) / 255.0
        depth = np.array(depth.resize((224,224)), dtype=np.float32) / 255.0
        depth = np.expand_dims(depth, axis=-1)

        # rgb_depth -> 4 channels
        x = np.expand_dims(np.concatenate([rgb, depth], axis=-1), axis=0)  # (1, 224, 224, 4) batch sjhape

        # predict
        prob = self.model.predict(x)[0][0]
        return bool(round(prob))  # (2)probability 추가(for test, SUCCESS (87.3%)) + (1) SUCCESS/FAIL