import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import re
import numpy as np

IMAGES_PATH = "Flicker8k_Dataset"
IMAGE_SIZE = (299, 299)
VOCAB_SIZE = 10000
SEQ_LENGTH = 25
EMBED_DIM = 512
FF_DIM = 512
BATCH_SIZE = 64

EPOCHS = 30


class DataLoader:
    def __init__(self,captions_path,img_path):
        self.captions_path=captions_path
        self.img_path=img_path
    
    def load_captions_data(self):
        with open(self.captions_path) as caption_file:
            caption_data = caption_file.readlines()
            caption_mapping = {}
            text_data = []
            images_to_skip = set()

            for line in caption_data:
                line = line.rstrip("\n")

                img_name, caption = line.split("\t")

                img_name = img_name.split("#")[0]
                img_name = os.path.join(self.img_path, img_name.strip())
                tokens = caption.strip().split()

                if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
                    images_to_skip.add(img_name)
                    continue

                if img_name.endswith("jpg") and img_name not in images_to_skip:
                    caption = "<start> " + caption.strip() + " <end>"
                    text_data.append(caption)

                    if img_name in caption_mapping:
                        caption_mapping[img_name].append(caption)
                    else:
                        caption_mapping[img_name] = [caption]

            for img_name in images_to_skip:
                if img_name in caption_mapping:
                    del caption_mapping[img_name]

            return caption_mapping, text_data
