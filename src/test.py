from model_durasi import make_emb_model
from tensorflow.keras.utils import plot_model

test = make_emb_model(128, 431)
print(plot_model(test, 'model.png'))
