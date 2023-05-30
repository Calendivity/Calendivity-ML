import tensorflow as tf
import numpy as np
from utils import *

class SequentialModel(tf.keras.Model):
  def __init__(self, num_features=10,emb_dim=64, 
                mask_value=-1,model_type='lstm', map_categorical=None):
    super(SequentialModel, self).__init__()
    # self.emb = make_emb_seq_model(emb_dim)
    if 'lstm' in model_type:
      self.mod = make_seq_model(num_features,emb_dim, mask_value)
    elif 'conv' in model_type:
      self.mod = make_conv1d_autoencoder(num_features, emb_dim, mask_value)
    self.mask_val = mask_value
    self.map_categorical = map_categorical

  def call(self, x):
    # x = x.to_tensor()
    padded_x = tf.pad(x, paddings=[[0,0], [0, 81 - tf.shape(x)[1]], [0,0]],
                      constant_values=self.mask_val)
    padded_x = tf.cast(padded_x, dtype=tf.float32)
    # x = self.emb(padded_x)
    x = self.mod(padded_x)
    return x

  def corrupt(self, x):
    # leng = int(x.shape[1])
    mask = tf.cast(x != self.mask_val, tf.int32)
    leng = tf.reduce_min(tf.reduce_sum(mask, axis=1))
    times = leng//5
    # a = tf.random.uniform(shape=[times], minval=0, maxval=leng, dtype=tf.int32)
    a = np.random.choice(leng, replace=False, size=times)
    b = tf.random.uniform(shape=(), minval=0, maxval=101, dtype=tf.double)

    for u in range(times):
      # corrupt_feat = tf.zeros_like(x[:,a[u]:a[u]+1, 2:3]) + b
      corrupt_feat = tf.zeros_like(
          x[:,a[u]:a[u]+1, 2:3]) + self.map_categorical["TRTIER2"][5001]
      cor_x = tf.concat([x[:,:a[u],2:3], corrupt_feat, x[:,a[u]+1:, 2:3]], axis=1)
    cor_x = tf.concat([x[:, :, :2], cor_x[:, :, :], x[:, :, 3:]], axis=-1)
    return cor_x, a, leng

  # @tf.function
  def shared_step(self, data):
    x = data.to_tensor(default_value=self.mask_val)
    cor_x, idx, leng = self.corrupt(x)
    padded_cor =tf.cast(cor_x, dtype=tf.float32)
    padded_x = tf.cast(x, dtype=tf.float32)
    real_x = x[:, :leng, :]
    cor_x = self(padded_cor)[:, :leng, :]
    loss = self.compiled_loss(real_x, cor_x)
    return loss

    
  def train_step(self, data):
    with tf.GradientTape() as tape:
      loss = self.shared_step(data)
    grads = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

    return {'reconstruction_loss' : loss}


  def validation_step(self, data):
    return {"reconstruction_loss" : self.shared_step(data)}
    
  def test_step(self,data):
    return {"reconstruction_loss" : self.shared_step(data)}

# seq_build_model = SequentialModel(10,32, -1, model_type='conv')