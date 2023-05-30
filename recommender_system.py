import tensorflow as tf
import numpy as np

user_cols = ["TUCASEID","TESEX", "PEEDUCA", "TRDTOCC1", "TRDPFTPT", "TESCHENR", "TEAGE"]
item_cols = ["TRTIER2"]
# edge_cols = ["TUCUMDUR24", "TUACTDUR", "TUSTARTTIM"]
edge_index = ["TUACTDUR", "TUSTARTTIM", "TUCUMDUR24"]

def make_emb_model_act(num_user_feat, num_item_feat, num_users, num_items, emb_dim):
  user_feat = tf.keras.Input((num_user_feat,), name='user_feat')
  item_feat = tf.keras.Input((num_item_feat,), name='item_feat')

  user_categ = tf.keras.layers.Embedding(num_users+1, emb_dim)(
        tf.keras.layers.Lambda(lambda x:x[:,:-1])(user_feat)
      )
  user_numeric = tf.keras.Sequential(
    [tf.keras.layers.Dense(2**n, activation='relu') for n in range(4, int(np.log2(emb_dim))+1)]
  )(tf.keras.layers.Lambda(lambda x:x[:, -1:])(user_feat))

  user_categ = tf.keras.layers.LSTM(emb_dim*2, return_sequences=True)(user_categ)
  user_categ = tf.keras.layers.GRU(emb_dim)(user_categ)
  user_emb = tf.keras.layers.Concatenate(axis=-1)([user_categ, user_feat])

  item_emb = tf.keras.layers.Embedding(num_items, emb_dim)(item_feat)
  item_emb = tf.keras.layers.LSTM(emb_dim*2, return_sequences=True)(item_emb)
  item_emb = tf.keras.layers.GRU(emb_dim)(item_emb)

  cat = tf.keras.layers.Concatenate(axis=-1)([user_emb, item_emb])

  cum=  tf.keras.layers.Dense(120, activation='relu')(cat)
  tim = tf.keras.layers.Dense(120, activation='relu')(
      tf.keras.layers.Concatenate(axis=-1)([cat, cum]))
  dur = tf.keras.layers.Dense(120, activation='relu')(
      tf.keras.layers.Concatenate(axis=-1)([cat, cum, tim]))
  cum=  tf.keras.layers.Dropout(0.2)(cum)
  tim=  tf.keras.layers.Dropout(0.2)(tim)
  dur=  tf.keras.layers.Dropout(0.2)(dur)

  dur=  tf.keras.layers.Dense(84, activation='relu')(
      tf.keras.layers.Concatenate(axis=-1)([dur,cum,tim]))
  cum = tf.keras.layers.Dense(84, activation='relu')(
      tf.keras.layers.Concatenate(axis=-1)([dur,cum,tim]))
  tim = tf.keras.layers.Dense(84, activation='relu')(
      tf.keras.layers.Concatenate(axis=-1)([dur,cum,tim]))
  cum=  tf.keras.layers.Dropout(0.2)(cum)
  tim=  tf.keras.layers.Dropout(0.2)(tim)
  dur=  tf.keras.layers.Dropout(0.2)(dur)

  tim=  tf.keras.layers.Dense(32, activation='relu')(
      tf.keras.layers.Concatenate(axis=-1)([dur,cum,tim]))
  dur = tf.keras.layers.Dense(32, activation='relu')(
      tf.keras.layers.Concatenate(axis=-1)([dur,cum,tim]))
  cum = tf.keras.layers.Dense(32, activation='relu')(
      tf.keras.layers.Concatenate(axis=-1)([dur,cum,tim]))
  cum=  tf.keras.layers.Dropout(0.2)(cum)
  tim=  tf.keras.layers.Dropout(0.2)(tim)
  dur=  tf.keras.layers.Dropout(0.2)(dur)

  cum=  tf.keras.layers.Dense(24, activation='relu')(cum)
  cum=  tf.keras.layers.Dense(16, activation='relu')(cum)
  tucumdur24=  tf.keras.layers.Dense(1, activation='relu', name='tucumdur24')(cum)

  tim=  tf.keras.layers.Dense(24, activation='relu')(tim)
  tim=  tf.keras.layers.Dense(16, activation='relu')(tim)
  tustarttim=  tf.keras.layers.Dense(1, activation='relu', name='tustarttim')(tim)

  dur=  tf.keras.layers.Dense(24, activation='relu')(dur)
  dur=  tf.keras.layers.Dense(16, activation='relu')(dur)
  tuactdur=  tf.keras.layers.Dense(1, activation='relu', name='tuactdur')(dur)

  return tf.keras.Model(inputs=[user_feat, item_feat], 
                        outputs=[tustarttim, tuactdur, tucumdur24])

# num_users = 10493
# num_items = 101
# num_user_feat = len(user_cols)
# num_item_feat = len(item_cols)
# emb_model_act = make_emb_model_act(num_user_feat, num_item_feat, num_users, num_items, 32)
# emb_model_act.summary()