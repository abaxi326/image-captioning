import keras
from keras import layers
import tensorflow as tf
from keras.applications import efficientnet

IMAGE_SIZE = (299, 299)
VOCAB_SIZE = 10000
SEQ_LENGTH = 25
EMBED_DIM = 512
FF_DIM = 512
BATCH_SIZE = 64
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE

class ImageCaptioningModel(keras.Model):
    def __init__(self,cnn_model,encoder,decoder,num_captions_per_image=5,image_aug=None):
        super().__init__()
        self.cnn_model=self.get_cnn_model()
        self.encoder=encoder
        self.decoder=decoder
        self.loss_tracker=keras.metrics.Mean(name='loss')
        self.accuracy_tracker=keras.metrics.Mean(name='accuracy')
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug
    
    def get_cnn_model():
        base_model = efficientnet.EfficientNetB0(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
   
        base_model.trainable = False
        base_model_out = base_model.output
        base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
        cnn_model = keras.models.Model(base_model.input, base_model_out)
        return cnn_model

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    
    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)
    
    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        encoder_out=self.encoder(img_embed,training=True)
        batch_seq_inp=batch_seq[:,:-1]
        batch_seq_true=batch_seq[:,1:]
        batch_seq_pred=self.decoder(batch_seq_inp,encoder_out,training=training)
        mask = tf.math.not_equal(batch_seq_true, 0)
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss,acc
    
    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        if self.image_aug:
            batch_img = self.image_aug(batch_img)

      
        img_embed = self.cnn_model(batch_img)

      
        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                loss, acc = self._compute_caption_loss_and_acc(
                    img_embed, batch_seq[:, i, :], training=True
                )

              
                batch_loss += loss
                batch_acc += acc

          
            train_vars = (
                self.encoder.trainable_variables + self.decoder.trainable_variables
            )

            
            grads = tape.gradient(loss, train_vars)

            
            self.optimizer.apply_gradients(zip(grads, train_vars))

       
        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_tracker.result(),
        }