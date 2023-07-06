import tensorflow as tf 
import numpy as np
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions

def make_model(emb_dim, num_classes):
    print("Initializing Model Durasi....")
    user_cols = ['TESEX', 'PEEDUCA', 'TRDTOCC1', 'TESCHENR', 'TRDPFTPT', 'TEAGE']
    edge_cols = ['TUSTARTTIM']
    item_cols = ['TRCODE']

    user_input = [tf.keras.Input((1,), name=name) for name in user_cols]
    edge_cols = [tf.keras.Input((1,), name=name) for name in edge_cols]
    item_cols = [tf.keras.Input((1,), name=name) for name in item_cols]

    categs = []
    te_age = None

    for x in user_input+item_cols:
        if x.name == 'TEAGE':
            te_age = x
        elif x.name == 'TRCODE':
            categs.append(
                tf.keras.layers.Embedding(num_classes, emb_dim, name='trcode_emb')(x)
            )
        # user_input.remove('TRCODE')
        elif x.name == 'PEEDUCA':
            categs.append(
                tf.keras.layers.Embedding(16, emb_dim, name='peeduca_emb')(x)
            )
        elif x.name == 'TESEX':
            categs.append(
                tf.keras.layers.Embedding(2, emb_dim, name='tesex_emb')(x)
            )
        elif x.name == 'TRDTOCC1':
            categs.append(
                tf.keras.layers.Embedding(23, emb_dim, name='trdtocc1_emb')(x)
            )
        elif x.name == 'TRDPFTPT':
            categs.append(
                tf.keras.layers.Embedding(3, emb_dim, name='trdpftpt_emb')(x)
            )
        elif x.name == 'TESCHENR':
            categs.append(
                tf.keras.layers.Embedding(3, emb_dim, name='teschenr_emb')(x)
            )
    categ_cols = tf.keras.layers.Concatenate(axis=-1)(categs)
    # categ_emb = tf.keras.layers.Embedding(101,emb_dim)(categ_cols)
    lstm = tf.keras.layers.LSTM(128, return_sequences=True)(categ_cols)
    lstm = tf.keras.layers.GRU(64)(lstm)

    concat = tf.keras.layers.Concatenate(axis=-1)([lstm, te_age, *edge_cols])

    lstm = tf.keras.Sequential([
        *[tf.keras.layers.Dense(2**n, activation='relu') for n in range(8,5,-1)]

    ])(concat)

    x = tf.keras.layers.Dense(120, activation='relu')(lstm)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(84, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    

    x = tf.keras.layers.Dense(1, activation='relu')(x)

    return tf.keras.Model(inputs=[*(user_input + item_cols + edge_cols)], outputs=x)

# emb_model = make_emb_model(128, 431)


def make_model2(emb_dim, num_classes):
  user_cols = ['TESEX', 'PEEDUCA', 'TRDTOCC1', 'TESCHENR', 'TRDPFTPT', 'TEAGE']
  edge_cols = ['TUSTARTTIM']
  item_cols = ['TRCODE']
  label_cols = ['TUACTDUR']

  user_input = [tf.keras.Input((1,), name=name) for name in user_cols]
  edge_cols = [tf.keras.Input((1,), name=name) for name in edge_cols]
  item_cols = [tf.keras.Input((1,), name=name) for name in item_cols]

  categs = []
  te_age = None

  for x in user_input+item_cols:
    if x.name == 'TEAGE':
      te_age = x
    elif x.name == 'TRCODE':
      categs.append(
          tf.keras.layers.Embedding(num_classes, emb_dim, name='trcode_emb')(x)
      )
      # user_input.remove('TRCODE')
    elif x.name == 'PEEDUCA':
      categs.append(
          tf.keras.layers.Embedding(16, emb_dim,name='peeduca_emb')(x)
      )
    elif x.name == 'TESEX':
      categs.append(
          tf.keras.layers.Embedding(2, emb_dim, name='tesex_emb')(x)
      )
    elif x.name == 'TRDTOCC1':
      categs.append(
          tf.keras.layers.Embedding(23, emb_dim, name='trdtocc1_emb')(x)
      )
    elif x.name == 'TRDPFTPT':
      categs.append(
          tf.keras.layers.Embedding(3, emb_dim, name='trdpftpt_emb')(x)
      )
    elif x.name == 'TESCHENR':
      categs.append(
          tf.keras.layers.Embedding(3, emb_dim,name='teschenr_emb')(x)
      )
  
  print(categs)
  print(te_age)
  
  categ_cols = tf.keras.layers.Concatenate(axis=1, name='categ_concat')(categs)
  # categ_emb = tf.keras.layers.Embedding(101,emb_dim)(categ_cols)
  lstm = tf.keras.layers.LSTM(emb_dim//2, return_sequences=True, name='categ_lstm')(categ_cols)
  lstm = tf.keras.layers.GRU(emb_dim//4, name='categ_gru')(lstm)

  numeric = tf.keras.layers.Concatenate(axis=-1, name='numeric_concat')([te_age, *edge_cols])
  print(np.log2(emb_dim))
  numeric = tf.keras.Sequential([
      *[tf.keras.layers.Dense(2**n, activation='relu') for n in range(3,int(np.log2(emb_dim))+1)]
  ], name='numeric_seq')(numeric)

  # print(te_age, x)
  concat = tf.keras.layers.Concatenate(axis=-1, name='concat_all')([lstm, numeric])

  lstm = tf.keras.Sequential([
      *[tf.keras.layers.Dense(2**n, activation='relu') for n in range(8,5,-1)]

  ], name='seq_all')(concat)

  u = tf.keras.layers.Dense(32, activation='relu', name='32_dense')(lstm)

  u = tf.keras.layers.Dense(1, name='last_l', activation='relu')(u)

  u = tf.keras.layers.Lambda(lambda m:tf.clip_by_value(m, 0, 1440), name='clip_val')(u)
  return tf.keras.Model(inputs=[*(user_input + item_cols + edge_cols)], outputs=u)


def make_emb_model(emb_dim, num_classes):
  user_cols = ['TESEX', 'PEEDUCA', 'TRDTOCC1', 'TESCHENR', 'TRDPFTPT', 'TEAGE']
  edge_cols = ['TUSTARTTIM']
  item_cols = ['TRCODE']

  user_input = [tf.keras.Input((1,), name=name) for name in user_cols]
  edge_cols = [tf.keras.Input((1,), name=name) for name in edge_cols]
  item_cols = [tf.keras.Input((1,), name=name) for name in item_cols]

  categs = []
  te_age = None

  for x in user_input+item_cols:
    if x.name == 'TEAGE':
      te_age = x
    elif x.name == 'TRCODE':
      categs.append(
          tf.keras.layers.Embedding(num_classes, emb_dim, name='trcode_emb')(x)
      )
      # user_input.remove('TRCODE')
    elif x.name == 'PEEDUCA':
      categs.append(
          tf.keras.layers.Embedding(16, emb_dim,name='peeduca_emb')(x)
      )
    elif x.name == 'TESEX':
      categs.append(
          tf.keras.layers.Embedding(2, emb_dim, name='tesex_emb')(x)
      )
    elif x.name == 'TRDTOCC1':
      categs.append(
          tf.keras.layers.Embedding(23, emb_dim, name='trdtocc1_emb')(x)
      )
    elif x.name == 'TRDPFTPT':
      categs.append(
          tf.keras.layers.Embedding(3, emb_dim, name='trdpftpt_emb')(x)
      )
    elif x.name == 'TESCHENR':
      categs.append(
          tf.keras.layers.Embedding(3, emb_dim,name='teschenr_emb')(x)
      )
  
  print(categs)
  print(te_age)
  
  categ_cols = tf.keras.layers.Concatenate(axis=1, name='categ_concat')(categs)

  lstm = tf.keras.layers.LSTM(emb_dim, return_sequences=True, name='categ_lstm')(categ_cols)
  lstm = tf.keras.layers.GRU(emb_dim//2, name='categ_gru')(lstm)

  numeric = tf.keras.layers.Concatenate(axis=-1, name='numeric_concat')([te_age, *edge_cols])
  print(np.log2(emb_dim))
  numeric = tf.keras.Sequential([
      *[tf.keras.layers.Dense(2**n, activation='relu') for n in range(4,int(np.log2(emb_dim//2)))]
  ], name='numeric_seq')(numeric)

  concat = tf.keras.layers.Concatenate(axis=-1, name='concat_all')([lstm, numeric])

  lstm = tf.keras.Sequential([
      *[tf.keras.layers.Dense(2**n, activation='relu') for n in range(8,5,-1)]

  ], name='seq_all')(concat)

  u = tf.keras.layers.Dense(32, activation='relu', name='32_dense')(lstm)

  u = tf.keras.layers.Dense(2, name='last_l', activation='elu')(u)
  u = tfpl.DistributionLambda(
      lambda x : tfd.Independent(
          tfd.Normal(x[:, :1], x[:, 1:]),
          reinterpreted_batch_ndims=1
      )
  )(u)

  return tf.keras.Model(inputs=[*(user_input + item_cols + edge_cols)], outputs=u)

    

