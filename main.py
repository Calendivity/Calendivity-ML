import tensorflow as tf
import pandas as pd
from data_utils import prepare_dataset
from utils import *

PATH_ATUS_2016 = 'data/atus_2016'
PATH_ATUS_2021 = 'data/atus_2021'


if '__name__' == '__main__':

    categorical_cols = ["TESEX", "PEEDUCA", "TRTIER2","TRDTOCC1", "TESCHENR", "TRDPFTPT"]
    numeric_cols = ["TUSTARTTIM", "TEAGE", "TUCUMDUR24"]

    user_cols = ["TUCASEID","TESEX", "PEEDUCA", "TRDTOCC1", "TRDPFTPT", "TESCHENR", "TEAGE"]
    item_cols = ["TRTIER2"]

    dataset = prepare_dataset(
        model_type='dur', path_dir=PATH_ATUS_2016,
        numeric_cols=numeric_cols, categorical_cols=categorical_cols, 
        user_cols=user_cols, item_cols=item_cols
    )
    model = get_model_dur(64, numeric_cols, categorical_cols)
