import tensorflow as tf

def normalize_numeric(dataframe, numeric_cols=['TUSTARTTIM', 'TEAGE'], 
                    scaler=None):

    df_new = dataframe.copy()
    df_new[numeric_cols] = scaler.transform(df_new[numeric_cols].to_numpy())
    
    return df_new


def normalize_categoric(dataframe, 
                        categorical_cols=["TESEX", "PEEDUCA","TRDTOCC1", 
                                        "TESCHENR", "TRDPFTPT", "TRCODE", "TUCASEID"], 
                        map_categorical=None):

    df_new = dataframe.copy()
    
    for categ in categorical_cols:
        df_new[categ] = df_new[categ].map(map_categorical[categ])
    
    return df_new


def remove_trcode_low_count(atusact, count_threshold=10, inplace=False):

    val_count = atusact['TRCODE'].value_counts()
    low_count =  val_count[val_count <= count_threshold].index.tolist()

    if inplace:
        atusact['TRCODE'] = atusact['TRCODE'].map(
            lambda x: -1 if x in low_count else x )
    else :
        atusact_new = atusact.copy()
        atusact_new['TRCODE'] = atusact_new['TRCODE'].map(
            lambda x: -1 if x in low_count or str(x)[:-2] == '5001' or x == 509999 else x )
        return atusact_new

def make_dataset(data, label_col='TUACTDUR', unused_cols=['TUCASEID'],
                one_hot=False, num_classes=None):
    if one_hot:
        features = tf.data.Dataset.from_tensor_slices(
        (
            dict(data.drop(columns=[label_col]+unused_cols, inplace=False)),
            tf.one_hot(data[label_col].values, depth=num_classes)
        )
    )
        
    else :
        features = tf.data.Dataset.from_tensor_slices(
            (
                dict(data.drop(columns=[label_col]+unused_cols, inplace=False)),
                data[label_col].astype(float).values
            )
        )

    return features