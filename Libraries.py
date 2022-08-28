#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
import seaborn as sns
import os
import random
import ast
# import keras
import platform
if platform.system() == "Windows":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import string
from functools import partial
from datetime import datetime
import matplotlib.pyplot as plt
import transformers as ppb
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
# from keras.preprocessing.text import Tokenizer
# from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix,roc_curve,roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval  # rand, GridSearch
from keras.models import Sequential, Model
from keras.initializers import Constant
from keras.layers import Dense, Input, Activation, Flatten, MaxPooling1D, Dropout, Conv1D, Concatenate, Embedding
from keras.layers import Bidirectional, LSTM, GRU, Attention, Layer, TextVectorization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow import convert_to_tensor
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix
from transformers import BertTokenizer, TFBertModel, BertConfig

seed = 99
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
# tf.keras.utils.set_random_seed #Equivalent to above 3

split_random_seed = 24 #Used for train/test split
SMOTE_random_seed = 58
KFold_shuffle_random_seed = 42 #Used for KFold split