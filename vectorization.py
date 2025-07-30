import tensorflow as tf
import re
import keras 
import numpy as np

IMAGE_SIZE = (299, 299)
VOCAB_SIZE = 10000
SEQ_LENGTH = 25
EMBED_DIM = 512
FF_DIM = 512
BATCH_SIZE = 64
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE

class TextImageVectorization:
    def __init__(self,caption_mapping,text_data):
        self.caption_mapping=caption_mapping
        self.text_data=text_data
        self.strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        self.strip_chars = self.strip_chars.replace("<", "")
        self.strip_chars = self.strip_chars.replace(">", "")
        self.vectorizer=keras.layers.TextVectorization(max_tokens=VOCAB_SIZE,
                        output_mode="int",
                        output_sequence_length=SEQ_LENGTH,
                        standardize=self.custom_standardization)
        self.vectorizer.adapt(text_data)
        self.image_augmentation= keras.Sequential(
                                    [
                                        keras.layers.RandomFlip("horizontal"),
                                        keras.layers.RandomRotation(0.2),
                                        keras.layers.RandomContrast(0.3),
                                    ]
                                )

    def custom_standardization(self,input_string):
        lowercase = tf.strings.lower(input_string)
        return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(self.strip_chars), "")
    
    def decode_and_resize(self,img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img
    
    def process_input(self,img_path, captions):
        return self.decode_and_resize(img_path), self.vectorizer(captions)
    
    def create_transformer_inputs(self):
        dataset=[self.process_input(i,j) for i,j in zip(self.caption_mapping.keys(),self.caption_mapping.values())]
        x,y=zip(*dataset)
        x=np.array(x)
        y=np.array(y)
        return x,y
