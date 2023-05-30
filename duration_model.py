import tensorflow as tf


def make_emb_model(num_cols_shape, categ_cols_shape, emb_dim):
  numeric_cols = tf.keras.Input((num_cols_shape,), name="numeric_feat")
  categ_cols = tf.keras.Input((categ_cols_shape,), name="categoric_feat")

  categs = []

  categs.append(
      tf.keras.layers.Embedding(101, emb_dim)(
        tf.keras.layers.Lambda(lambda x:x[:,2])(categ_cols)
      )
  )

  categs.append(
      tf.keras.layers.Embedding(16, emb_dim)(
        tf.keras.layers.Lambda(lambda x:x[:,1])(categ_cols)
      )
  )

  categs.append(
      tf.keras.layers.Embedding(2, emb_dim)(
        tf.keras.layers.Lambda(lambda x:x[:,0])(categ_cols)
      )
  )

  categs.append(
      tf.keras.layers.Embedding(23, emb_dim)(
        tf.keras.layers.Lambda(lambda x:x[:,3])(categ_cols)
      )
  )

  categs.append(
      tf.keras.layers.Embedding(3, emb_dim)(
        tf.keras.layers.Lambda(lambda x:x[:,5])(categ_cols)
      )
  )

  categs.append(
      tf.keras.layers.Embedding(3, emb_dim)(
        tf.keras.layers.Lambda(lambda x:x[:,4])(categ_cols)
      )
  )

  concat = tf.keras.layers.Concatenate(axis=-1)([*categs, numeric_cols])

  dense = tf.keras.Sequential([
      *[tf.keras.layers.Dense(2**n, activation='relu') for n in range(8,5,-1)]

  ])(concat)

  x = tf.keras.layers.Dense(120, activation='relu')(dense)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(84, activation='relu')(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(32, activation='relu')(x)
  x = tf.keras.layers.Dense(1, activation='relu')(x)

  return tf.keras.Model(inputs=[numeric_cols, categ_cols], outputs=x)

# num_cols_shape = len(numeric_cols)
# categ_cols_shape = len(categorical_cols)
# emb_model = make_emb_model(num_cols_shape, categ_cols_shape, 64)
# emb_model.summary()