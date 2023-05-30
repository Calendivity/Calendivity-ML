import tensorflow as tf
import pandas as pd
import os 
from sklearn import preprocessing

def prepare_dataset(model_type, path_dir, numeric_cols, categorical_cols,
                    user_cols, item_cols):

    # categorical_cols = ["TESEX", "PEEDUCA", "TRTIER2","TRDTOCC1", "TESCHENR", "TRDPFTPT"]
    # numeric_cols = ["TUSTARTTIM", "TEAGE", "TUCUMDUR24"]

    dataset = load_data(path_dir)
    mappings, dataset = encode_categorical(dataset, categorical_cols)

    if 'seq' in model_type:
        return seq_dataset(dataset, numeric_cols, categorical_cols)

    elif 'dur' in model_type:
        return dur_dataset(dataset, numeric_cols, categorical_cols)

    else :
        return recommender_dataset(dataset)
        

def seq_dataset(dataset, numeric_cols, categorical_cols):
    scalar, dataset = normalize(dataset, numeric_cols)

    dataset_sorted = dataset.sort_values(by=['TUCASEID','TUCUMDUR24'])

    map_unique = map_id_unique_value(
        dataset.drop(columns='TUCASEID', inplace=False), categorical_cols
    )

    map_emb = {
        x:dataset_sorted.loc[dataset_sorted['TUCASEID']==x].drop(
            columns='TUCASEID', inplace=False
            ).values for x in dataset_sorted['TUCASEID'].unique()
    }

    emb_seq = tf.ragged.constant(list(map_emb.values()))
    return tf.data.Dataset.from_tensor_slices(emb_seq), scalar, map_unique
    

def recommender_dataset(dataset, user_cols, item_cols):
    # user_cols = ["TUCASEID","TESEX", "PEEDUCA", "TRDTOCC1", "TRDPFTPT", "TESCHENR", "TEAGE"]
    # item_cols = ["TRTIER2"]
    
    feat_dataset = tf.data.Dataset.from_tensor_slices(
        {"user_feat":dataset[user_cols].values, 
        "item_feat":dataset[item_cols].values}
    )

    label_dataset = tf.data.Dataset.from_tensor_slices(
        {'tustarttim':dataset['TUSTARTTIM'].values, 
        "tuactdur":dataset['TUACTDUR'].values, 
        "tucumdur24":dataset['TUCUMDUR24'].values}
    )

    return tf.data.Dataset.zip((feat_dataset, label_dataset))


def dur_dataset(dataset, numeric_cols, categorical_cols):
    label_col = "TUACTDUR"
    scalar, dataset = normalize(dataset, numeric_cols)
    feature_dataset = tf.data.Dataset.from_tensor_slices(
        {"numeric_feat":dataset[numeric_cols].values,
        "categoric_feat":dataset[categorical_cols].values}
    )
    label_dataset = tf.data.Dataset.from_tensor_slices(
        dataset[label_col].values
    )
    return tf.data.Dataset.zip((feature_dataset, label_dataset)), scalar



def map_id_unique_value(dataset, columns):
    return {
        x:(i, len(dataset[x].unique())) 
        for i,x in enumerate(dataset.columns.values)
        if x in columns
    }
        

def encode_categorical(dataset, categorical_features, inplace=False):
    map_categorical = {}

    if inplace :
        for x in categorical_features:
            m_unique = dataset[x].unique()
            m_map = {x:y for x,y in zip(m_unique, range(len(m_unique)))}
            dataset[x] = dataset[x].map(m_map)
            map_categorical[x] = m_map
        return map_categorical
    else :
        dataset2 = dataset.copy()
        for x in categorical_features:
            m_unique = dataset2[x].unique()
            m_map = {x:y for x,y in zip(m_unique, range(len(m_unique)))}
            dataset2[x] = dataset2[x].map(m_map)
            map_categorical[x] = m_map
        return map_categorical, dataset2

def normalize(dataset, numerical_features, inplace=False):
    if inplace:
        scaler = preprocessing.StandardScaler().fit(dataset[numerical_features].values)
        dataset[numerical_features] = scaler.transform(dataset[numerical_features].values)
        return scaler
    else :
        dataset2 = dataset.copy()
        scaler = preprocessing.StandardScaler().fit(dataset2[numerical_features].values)
        dataset2[numerical_features] = scaler.transform(dataset2[numerical_features].values)
        return scaler, dataset2
    


def load_data(path_dir):
    atusresp = pd.read_csv(os.path.join(path_dir,'atusresp_2016.dat'))
    atussum = pd.read_csv(os.path.join(path_dir,'atussum_2016.dat'))
    atusact  = pd.read_csv(os.path.join(path_dir,'atusact_2016.dat'))
    
    atusact_col = ["TUCASEID", "TUSTARTTIM", "TUACTDUR", "TRTIER2","TUCUMDUR24"] # TEWHERE
    atussum_col = ["TUCASEID","TEAGE", "TESEX", "PEEDUCA"] #+ [x for x in atussum.columns if x[0] == 't']
    atusresp_col = ["TUCASEID", "TRDTOCC1", "TESCHENR", "TRDPFTPT"] 
    combined = convert_tustarttim(atusact[atusact_col].merge(
        atusresp[atusresp_col].merge(
            atussum[atussum_col], on=['TUCASEID'], validate='1:1', how='inner')
        , on=['TUCASEID'], how='inner', validate="m:1")
    )

    return convert_tustarttim(combined)



# ubah kolom string TUSTARTTIM menjadi integer untuk merepresentasikan jumlah menit sejak jam 00:00
def convert_tustarttim(dataframe, inplace=False):
  if inplace == False:
    new_df = dataframe.copy(deep=True)
    new_df['TUSTARTTIM'] = new_df['TUSTARTTIM'].map(lambda x : int(x[:2]) * 60 + int(x[3:5]) )
    return new_df
  else :
    dataframe['TUSTARTTIM'] = dataframe['TUSTARTTIM'].map(lambda x : int(x[:2]) * 60 + int(x[3:5]) )


# ubah kolom string TUSTOPTIME menjadi integer untuk merepresentasikan jumlah menit sejak jam 00:00
def convert_tustoptime(dataframe, inplace=False):
  if inplace == False:
    new_df = dataframe.copy(deep=True)
    new_df['TUSTOPTIME'] = new_df['TUSTOPTIME'].map(lambda x : int(x[:2]) * 60 + int(x[3:5]) )
    return new_df
  else :
    dataframe['TUSTOPTIME'] = dataframe['TUSTOPTIME'].map(lambda x : int(x[:2]) * 60 + int(x[3:5]) )
    