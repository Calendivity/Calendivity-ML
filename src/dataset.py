import torch
import torch.nn.functional as F
from utils import *

def model_embedding_dataset(map_code, map_categorical, tokenizer):
    feature_data = []
    label_data = []
    for k,v in map_code.items():
        if k in map_categorical['TRCODE'].keys():
            for x in v:
                feature_data.append(x)
                label_data.append(map_categorical['TRCODE'][k])
        else :
            for x in v:
                feature_data.append(x)
                label_data.append(map_categorical['TRCODE'][-1])

    features = tokenizer(feature_data, max_length=30, padding='max_length', return_tensors='pt')

    label_tensor = F.one_hot(torch.tensor(label_data))
    feature_tensor = torch.dstack([
            features['input_ids'], 
            features['token_type_ids'], 
            features['attention_mask']
        ])

    emb_dataset = torch.utils.data.TensorDataset(
        feature_tensor,
        label_tensor
    )

    dataloader = torch.utils.data.DataLoader(emb_dataset, batch_size=64, shuffle=True)

    return dataloader



def split_train_test(dataset, len_data, batch_size, test_size=0.01):
    test_count = int(len_data * test_size)
    shuffled = dataset.shuffle(len_data)
    train = shuffled.skip(test_count)
    test = shuffled.take(test_count)
    return train.batch(batch_size), test.batch(batch_size)


def model_durasi_dataset(data, count_threshold, numerical_cols, 
                    categorical_cols, label_col, unused_cols, one_hot, num_classes):
    data = remove_trcode_low_count(data, count_threshold)
    data = normalize_numeric(data,  numerical_cols)
    data = normalize_categoric(data, categorical_cols)
    data = make_dataset(data, label_col, unused_cols, one_hot, num_classes)
    return data
        