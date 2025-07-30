from data_loader import DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt
from vectorization import TextImageVectorization

IMAGES_PATH = "C:\\Users\\a3318\\Downloads\\Flickr8k_Dataset\\Flicker8k_Dataset"
IMAGE_SIZE = (299, 299)
VOCAB_SIZE = 10000
SEQ_LENGTH = 25
EMBED_DIM = 512
FF_DIM = 512
BATCH_SIZE = 64
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE

def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

captions_path='C:\\Users\\a3318\\OneDrive - Axtria\\Documents\\Codes\\Flickr8k_text\\Flickr8k.token.txt'
img_path='C:\\Users\\a3318\\Downloads\\Flickr8k_Dataset\\Flicker8k_Dataset'

data_loader1=DataLoader(captions_path,img_path)
caption_mapping,text_data=data_loader1.load_captions_data()

vectorizer=TextImageVectorization(caption_mapping,text_data)

x,y=vectorizer.create_transformer_inputs()
print(x.shape,y.shape)


