import tensorflow as tf
import sequential_model, duration_model, recommender_system
import pickle, os

def make_seq_model(num_features=10,emb_dim = 64, mask_val=-1):
  
  feat_shape = emb_dim*6 + 4

  num_idx = [0,1,3,7]
  nums = []
  categs =[]
  inp = tf.keras.Input((81,10))
  input = tf.keras.layers.Masking(mask_val)(inp)
  

  categs.append(tf.keras.layers.Embedding(
      input_dim=2, 
      output_dim=emb_dim, 
      name=f'TESEX_emb')(tf.keras.layers.Lambda(lambda x: x[:, :, 8])(input)))
  categs.append(tf.keras.layers.Embedding(
      input_dim=23, 
      output_dim=emb_dim, 
      name=f'TRDTOCC1_emb')(tf.keras.layers.Lambda(lambda x: x[:, :, 4])(input)))
  categs.append(tf.keras.layers.Embedding(
      input_dim=3, 
      output_dim=emb_dim, 
      name=f'TESCHENR_emb')(tf.keras.layers.Lambda(lambda x: x[:, :, 5])(input)))
  categs.append(tf.keras.layers.Embedding(
      input_dim=3, 
      output_dim=emb_dim, 
      name=f'TRDPFTPT_emb')(tf.keras.layers.Lambda(lambda x: x[:, :, 6])(input)))
  categs.append(tf.keras.layers.Embedding(
      input_dim=16, 
      output_dim=emb_dim, 
      name=f'PEEDUCA_emb')(tf.keras.layers.Lambda(lambda x: x[:, :, 9])(input)))
  categs.append(tf.keras.layers.Embedding(
      input_dim=101, 
      output_dim=emb_dim, 
      name=f'TRTIER2_emb')(tf.keras.layers.Lambda(lambda x: x[:, :, 2])(input)))
  
  for i in num_idx:
    input_num = tf.keras.layers.Lambda(lambda x:x[:, :, i:i+1])(input)
    nums.append(input_num)
  

  num_feat = tf.keras.layers.Concatenate(axis=-1)(nums)
  num_mask = tf.keras.layers.Masking(mask_val)(num_feat)
  num_comb = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(8))(num_mask)
  cat_embs = tf.keras.layers.Concatenate(axis=-1)(categs)
  concat = tf.keras.layers.Concatenate(axis=-1)([num_comb,cat_embs])

  seq_model = tf.keras.Sequential([
    # tf.keras.layers.InputLayer((81,feat_shape)),
    # tf.keras.layers.Masking(mask_val),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True), merge_mode='sum'
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True), merge_mode='sum'
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True), merge_mode='sum'
    ),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(num_features, return_sequences=True), merge_mode='sum'
    ),
  ])(concat)

  return tf.keras.Model(inputs=inp, outputs=seq_model)

# emb_seq_model = make_emb_seq_model()
# seq_model_final = make_seq_model()

# emb_seq = tf.ragged.constant(list(map_emb.values()))
# dataset = tf.data.Dataset.from_tensor_slices(emb_seq)
# seq_model_final.summary()



def make_conv1d_autoencoder(num_features=10, emb_dim =32, mask_val=-1):
  num_idx = [0,1,3,7]
  nums = []
  categs =[]
  inp = tf.keras.Input((81,10))
  input = tf.keras.layers.Masking(mask_val)(inp)
  

  categs.append(tf.keras.layers.Embedding(
      input_dim=2, 
      output_dim=emb_dim, 
      # mask_zero=True,
      name=f'TESEX_emb')(tf.keras.layers.Lambda(lambda x: x[:, :, 8])(input)))
  categs.append(tf.keras.layers.Embedding(
      input_dim=23, 
      output_dim=emb_dim, 
      name=f'TRDTOCC1_emb')(tf.keras.layers.Lambda(lambda x: x[:, :, 4])(input)))
  categs.append(tf.keras.layers.Embedding(
      input_dim=3, 
      output_dim=emb_dim, 
      name=f'TESCHENR_emb')(tf.keras.layers.Lambda(lambda x: x[:, :, 5])(input)))
  categs.append(tf.keras.layers.Embedding(
      input_dim=3, 
      output_dim=emb_dim, 
      name=f'TRDPFTPT_emb')(tf.keras.layers.Lambda(lambda x: x[:, :, 6])(input)))
  categs.append(tf.keras.layers.Embedding(
      input_dim=16, 
      output_dim=emb_dim, 
      name=f'PEEDUCA_emb')(tf.keras.layers.Lambda(lambda x: x[:, :, 9])(input)))
  categs.append(tf.keras.layers.Embedding(
      input_dim=101, 
      output_dim=emb_dim, 
      name=f'TRTIER2_emb')(tf.keras.layers.Lambda(lambda x: x[:, :, 2])(input)))
  
  for i in num_idx:
    input_num = tf.keras.layers.Lambda(lambda x:x[:, :, i:i+1])(input)
    nums.append(input_num)
  

  num_feat = tf.keras.layers.Concatenate(axis=-1)(nums)
  num_mask = tf.keras.layers.Masking(mask_val)(num_feat)
  num_comb = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(8))(num_mask)
  cat_embs = tf.keras.layers.Concatenate(axis=-1)(categs)
  concat = tf.keras.layers.Concatenate(axis=-1)([num_comb,cat_embs])

  lstm = tf.keras.layers.GRU(10, return_sequences=True)(concat)

  enc= tf.keras.Sequential([
      tf.keras.layers.Conv1D(2**n, 3, padding='same', activation='relu') for n in range(5,9)
  ])(lstm)
  dec = tf.keras.Sequential([
      tf.keras.layers.Conv1DTranspose(2**n, 3, padding='same', 
                                      activation='relu') for n in range(8,4,-1)
  ])(enc)

  last = tf.keras.layers.Conv1D(num_features, 3, padding='same')(dec)

  return tf.keras.Model(inp, last)



def save_model(model, path_dir):
    model.save_weights(path_dir+"/weights")
    tf.saved_model(model, path_dir)

def load_model(model, path_dir):
    model.load_weights(path_dir+"/weights")

def save_pickle(pickle, path_dir, file_name=pickle.pkl):
    with open(os.path.join(path_dir, file_name), 'wb') as f:
        pickle.dump(pickle, f)
        print(f"Pickle stored in {os.path.join(path_dir, file_name)}")

def load_pickle(pickle, path_dir, file_name):
    loaded = None
    with open(os.path.join(path_dir, file_name), 'rb') as f:
        loaded = pickle.load(pickle, f)
    return loaded


def get_model_seq(emb_dim, model_type='conv'):
    return sequential_model.SequentialModel(10,emb_dim, -1, model_type=model_type)

def get_model_dur(emb_dim, numerical_cols, categorical_cols):
    num_cols_shape = len(numerical_cols)
    categ_cols_shape = len(categorical_cols)
    return duration_model.make_emb_model(num_cols_shape, categ_cols_shape, emb_dim)

def get_model_recommender(emb_dim,user_cols, 
                            item_cols, num_users, num_items):
    num_user_feat = len(user_cols)
    num_item_feat = len(item_cols)
    return recommender_system.make_emb_model_act(
        num_user_feat, num_item_feat, num_users, num_items, emb_dim)
