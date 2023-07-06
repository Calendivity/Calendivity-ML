import pickle
# from transformers import AutoTokenizer
import model_embedding 
import model_durasi 
from dataset import *
from flask import Flask, request
import pandas as pd
import pathlib
import numpy as np
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sentencepiece
from model_embedding import EmbModel



app = Flask(__name__)

def init_model(tokenizer_dir, pretrained_model_dir,
                 durasi_model_weights_dir, emb_model_weights_dir,
                 model_translate_dir, tokenizer_translate_dir):
    print("Initializing Model...")

    durasi_model = model_durasi.make_emb_model(128, 431)
    durasi_model.load_weights(durasi_model_weights_dir)
    # durasi_model = tf.keras.models.load_model(durasi_model_weights_dir)
    print("model_durasi has been succesfully initialized")

    emb_model = model_embedding.EmbModel(431, pretrained_model_dir)
    emb_model.load_state_dict(torch.load(emb_model_weights_dir))
    print("model_embedding has been succesfully initialized")
    
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer = torch.load(tokenizer_dir)
    print("tokenizer has been succesfully initialized")
    
    # model_translate = AutoModelForSeq2SeqLM.from_pretrained(model_translate_dir)
    # tokenizer_translate = AutoTokenizer.from_pretrained(tokenizer_translate_dir)
    # print("Model & Tokenizer translate has been successfully initialized")

    return durasi_model, emb_model, tokenizer#, model_translate, tokenizer_translate

def init_utils(scaler_dir, map_dir, map_code_dir):
    print("Initializing Utils...")
    with open(f'{map_dir}', 'rb') as f:
        map_categorical = pickle.load(f)
    with open(f'{scaler_dir}', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{map_code_dir}', 'rb') as f:
        map_code = pickle.load(f)
    print("all utilities succesfully initialized")
    return scaler, map_categorical, map_code


d = pathlib.Path().resolve().parent

ROOT_DIR = "https://storage.googleapis.com/capstone-project-386814-ml-bucket"

# TOKENIZER_DIR = d/"weights"/"tokenizer"
# TOKENIZER_DIR = d/"models"/"tokenizer.pth"
# # PRETRAINED_DIR =d/"weights"/"huggingface_model"
# PRETRAINED_DIR = d/"models"/"huggingface_model.pth"
# MODEL_DURASI_DIR = d/"weights"
# MODEL_DURASI_DIR = d/"weights"/"weights"
# MODEL_DURASI_DIR = d/"weights"/"model_durasi"/"weights"
# MODEL_DURASI_DIR = d/"weights"/"dist_weight"/"weights"
# # MODEL_DURASI_DIR = d/"models"/"model_durasi.h5"
# MODEL_EMB_DIR =d/"weights"/"model_embedding"/"model_weights.pth"
# # MODEL_EMB_DIR = d/"models"/"model_embedding.pth"
# MODEL_TRANSLATE_DIR = d/"weights"/"model_translate"
# TOKENIZER_TRANSLATE_DIR = d/"weights"/"tokenizer_translate"

# SCALER_DIR = d/ "utils"/"scaler.pkl"
# MAP_CODE_DIR = d/ "utils"/"map_code.pkl"
# MAP_DIR = d / "utils"/"map.pkl"

TOKENIZER_DIR = ROOT_DIR+"/weights/tokenizer"
PRETRAINED_DIR =ROOT_DIR+"/weights/huggingface_model"
MODEL_DURASI_DIR = ROOT_DIR+'/weights/dist_weight/weights'
MODEL_EMB_DIR =ROOT_DIR+'/weights/model_embedding/model_weights.pth'
MODEL_TRANSLATE_DIR = ROOT_DIR+"/weights/model_translate"
TOKENIZER_TRANSLATE_DIR = ROOT_DIR+"/weights/tokenizer_translate"

SCALER_DIR = ROOT_DIR+"/utils/scaler.pkl"
MAP_CODE_DIR = ROOT_DIR+"/utils/map_code.pkl"
MAP_DIR = ROOT_DIR+"/utils/map.pkl"

# url = "https://storage.googleapis.com/capstone-project-386814-ml-bucket/utils.tar"
# local_dir = tf.keras.utils.get_file("weights", origin=url, cache_dir='../tmp', untar=True)
# print(local_dir)


scaler, map_categorical, map_code = init_utils(
    SCALER_DIR, MAP_DIR, MAP_CODE_DIR
)
inv_map_trcode = {v:k for k,v in map_categorical['TRCODE'].items()}

# durasi_model, emb_model, tokenizer, model_translate, tokenizer_translate = init_model(
#     TOKENIZER_DIR, PRETRAINED_DIR, MODEL_DURASI_DIR, MODEL_EMB_DIR,
#     MODEL_TRANSLATE_DIR, TOKENIZER_TRANSLATE_DIR
# )

durasi_model, emb_model, tokenizer= init_model(
    TOKENIZER_DIR, PRETRAINED_DIR, MODEL_DURASI_DIR, MODEL_EMB_DIR,
    MODEL_TRANSLATE_DIR, TOKENIZER_TRANSLATE_DIR
)

print(durasi_model.summary())

print("\n\n\n")

print()


# def pipeline_translate(model, tokenizer, sentence):
#     translated = model.generate(
#         **tokenizer(sentence, return_tensors="pt", truncation=True, 
#                     padding='max_length', max_length=30))
#     output_sentence = [
#         tokenizer.decode(t, skip_special_tokens=True) for t in translated]
#     return output_sentence[0]


numerical_cols = ["TEAGE", "TUSTARTTIM"]
categorical_cols = ["TRCODE", "PEEDUCA", "TESCHENR","TESEX", "TRDPFTPT", "TRDTOCC1"]
columns = ['age', 'lastEducation', 'job', 'gender', 'education', 
            'employmentType', 'activityName', 'startTime', 'duration']
map_cols = {
    'age' : 'TEAGE',
    'lastEducation' : 'PEEDUCA',
    'job' : 'TRDTOCC1',
    'gender':'TESEX',
    'employmentType' : 'TRDPFTPT',
    'education' : 'TESCHENR',
    'activityName' : 'TRCODE',
    'startTime' : 'TUSTARTTIM',
    'duration' : 'TUACTDUR'
}

def preprocess_dataframe_from_api(api_type):
    df = {}
    for k,v in map_cols.items():
        if k =='duration' or ('act' not in api_type and k[:3]=='act'):
            continue
        print(request.args.get(k))
        if k in categorical_cols:
            df[v] = [int(request.args.get(k))]
        else :
            df[v] = [request.args.get(k)]

    df = pd.DataFrame.from_dict(df)
    acts = None
    if 'act' not in api_type :
        # acts = pipeline_translate(model_translate, tokenizer_translate, 
        #                             str(request.args.get('activityName')))
        acts = str(request.args.get('activityName'))
        df['TRCODE'] = emb_model.pipeline(tokenizer, acts).numpy()
    else :
        df['TRCODE'] = 0
    df = df.astype('int32')
    for x in categorical_cols:
        if x == 'TRCODE':
            continue
        df[x] = df[x].map(map_categorical[x])
    
    df[numerical_cols] = scaler.transform(df[numerical_cols].to_numpy())
    return df, acts


@app.route('/difficulty')
def home():
    df, acts = preprocess_dataframe_from_api("diff")

    dataset = tf.data.Dataset.from_tensor_slices(
        dict(df.astype(float))
    )

    real_dur = int(request.args.get('duration'))
    mod = durasi_model([x for x in dataset.batch(1).take(1)][0])
    pred_dur = np.abs(mod.sample(100000).numpy().flatten())
    pred_dur = pred_dur[np.argmin(np.abs(pred_dur - real_dur))]
    
    selisih = real_dur - pred_dur
    if pred_dur < 30//2:
        tingkat_kesulitan = 1
    elif pred_dur < 30:
        tingkat_kesulitan = 2
    elif selisih < 0 :
        if real_dur <= 30:
            tingkat_kesulitan = 2
        else :
            tingkat_kesulitan = 3
    elif selisih <=  30:
        tingkat_kesulitan = 3
    elif selisih <= 30*2:
        tingkat_kesulitan = 4
    else :
        tingkat_kesulitan = 5

    bobot_exp = tf.sigmoid(tf.nn.relu(real_dur-selisih)).numpy()
    exp = (
        tingkat_kesulitan * 7 * bobot_exp if tingkat_kesulitan <= 2
        else tingkat_kesulitan * 5 * bobot_exp
    )
    exp = 1 if exp < 1 else int(exp)

    print("Selisih :", selisih)

    print(tingkat_kesulitan, exp, inv_map_trcode[df['TRCODE'].values[0]])
    return {
        'difficulty':tingkat_kesulitan,
        'exp' : exp,
        'translation' : acts,
        'pred_dur' : str(pred_dur),
        'real_dur' : str(real_dur)
        }    
    
    

@app.route('/activity')
def act():
    df,_ = preprocess_dataframe_from_api(api_type='activity')

    df = pd.concat([df for _ in range(len(map_categorical['TRCODE'].values()))], ignore_index=True)
    df['TRCODE'] = np.array(list(map_categorical['TRCODE'].values() )).flatten()
    
    dataset = tf.data.Dataset.from_tensor_slices(
            dict(df.astype(float))
    )

    real_dur = int(request.args.get('duration'))
    mod = durasi_model([x for x in dataset.batch(len(df)).take(1)][0])
    pred_dur = mod.sample(100).numpy().reshape(431, 100)
    argmin= np.argmin(np.abs(pred_dur - real_dur), axis=1)
    preds = []
    for i,x in enumerate(argmin):
        preds.append(pred_dur[i,x])
    pred_dur = np.array(preds)
    pred_dur = pred_dur[argmin]
    
    df['pred'] = pred_dur.flatten()
    df['TUACTDUR'] = int(request.args.get('duration'))
    df['selisih'] = np.abs(df['pred'] - df['TUACTDUR'])
    idx = np.argsort(df['selisih'].values)
    val = df['TRCODE'].iloc[idx[:10]].values

    val = [
        map_code[inv_map_trcode[x]] if inv_map_trcode[x] != -1 
        else ["Free Time", "Relaxing"] for x in val
        ]
    val = [x[0].replace(", n.e.c.*", "") for x in val]
    return {x:y for x,y in enumerate(val)}
    

# new_model = EmbModel(431, 384)
if __name__ == '__main__':
    app.run(debug=True)

