#!/usr/bin/env python3

from Libraries import *
from StatisticsFunctions import multiclass_ROC_AUC
# from EmbeddingFunctions import getTokens

class E2ESentenceTransformer(tf.keras.Model):
    def __init__(self, model_name_or_path, num_labels, **kwargs):
        super().__init__()
        # loads the in-graph tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        # loads our TFSentenceTransformer
        self.model = TFSentenceTransformer(model_name_or_path, from_pt=True)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(num_labels, activation="softmax")

    def call(self, inputs):
        # runs tokenization and creates embedding
        # tokenized = self.tokenizer(inputs)
        embeddings = self.model(inputs)
        dropout_emb = self.dropout(embeddings)
        return self.classifier(dropout_emb)


    # def model(self):
    #     x = Input(shape=(24, 24, 3))
    #     return Model(inputs=[x], outputs=self.call(x))

# https://www.philschmid.de/tensorflow-sentence-transformers
class TFSentenceTransformer(tf.keras.layers.Layer):
    def __init__(self, model_name_or_path, **kwargs):
        super(TFSentenceTransformer, self).__init__()
        # loads transformers model
        self.model = TFAutoModel.from_pretrained(model_name_or_path, from_pt=True)

    def call(self, inputs, normalize=True):
        # runs model on inputs
        model_output = self.model(inputs)
        # Perform pooling. In this case, mean pooling.
        embeddings = self.mean_pooling(model_output, inputs["attention_mask"])
        # normalizes the embeddings if wanted
        if normalize:
          embeddings = self.normalize(embeddings)

        embeddings = embeddings

        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = tf.cast(
            tf.broadcast_to(tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)),
            tf.float32
        )
        return tf.math.reduce_sum(token_embeddings * input_mask_expanded, axis=1) / tf.clip_by_value(tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max)

    def normalize(self, embeddings):
      embeddings, _ = tf.linalg.normalize(embeddings, 2, axis=1)
      return embeddings

# Fully Dense Neural Network
def denseNN(param_grid, Xtrain, ytrain, X_test, y_test, num_labels):
    """
    parameters
    ------------------
    num_labels
        number of labels
    """

    # Fully Dense Network
    dense = Sequential()

    dense_path = "dense"

    dense = Sequential()
    dense.add(Input(shape=(768,)))
    dense.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    dense.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    dense.add(Dense(num_labels, activation='softmax'))

    dense.compile(optimizer='adam',
                  # loss='categorical_crossentropy',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    mc = ModelCheckpoint(dense_path + ".h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    dense.fit(Xtrain, ytrain, epochs=param_grid["epochs"], batch_size=param_grid["batch_size"], verbose=1)
    loss, accuracy = dense.evaluate(X_test, y_test, verbose=2)
    dense.summary()
    return loss

def objectiveFunctionCNN(param_grid, Xtrain, ytrain, Xtest, ytest, num_channels, num_features, num_label, modelType, e_type, know_infus_bool,
                         es, mc, max_length = 512, vocabSize = [], embedding_matrix = []):

    model = cnnModel(param_grid, num_channels, num_features, num_label, e_type, know_infus_bool, max_length, vocabSize, embedding_matrix)
    model.fit(Xtrain, ytrain, validation_data=(Xtrain, ytrain), epochs=param_grid["epochs"], batch_size=param_grid["batch_size"], callbacks=[es, mc], verbose=0)
    loss, accuracy, auc_values = model.evaluate(Xtest, ytest, verbose=2)
    return loss


def objectiveFunctionRNN(param_grid, Xtrain, ytrain, Xtest, ytest, num_channels, num_features, n_lab, model_name, modelType, e_type, emb_dim,
                         know_infus_bool, es, mc, preTrainDim = 300, max_length = 512, vocabSize = [], embedding_matrix = []):

    model = RNNModel(param_grid, num_channels, num_features, n_lab, e_type, model_name, modelType, know_infus_bool, emb_dim, preTrainDim, max_length, vocabSize, embedding_matrix)
    model.fit(Xtrain, ytrain, validation_data=(Xtrain, ytrain), epochs=param_grid["epochs"], batch_size=param_grid["batch_size"], callbacks=[es, mc], verbose=0)
    loss, accuracy, auc_values = model.evaluate(Xtest, ytest, verbose=2)
    return loss

#Convolutional Network
def cnnModel(param_grid, num_channels, num_features, num_label, e_type, know_infus_bool, max_length = 512, vocabSize = [], embedding_matrix =[]):
    """
    parameters
    ------------------
    num_channels
        maximum length of sentences
    num_features
        number of features
    num_labels
        number of labels
    """
    cnn_path = "CNN"
    metrics = [tf.keras.metrics.CategoricalAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')]
    if num_label == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'

    filters = 100

    if e_type == "BERT" and know_infus_bool == True:
        isa_index, aff_index = 0, 1
    elif e_type == "ConceptNet" and know_infus_bool == True:
        con_index, isa_index, aff_index = 0, 1, 2
    elif e_type == "ConceptNet" and know_infus_bool == False:
        con_index = 0

    if e_type == "BERT":
        input_shape = Input(shape=(num_channels, num_features))
        lane_1 = Conv1D(filters=filters, kernel_size=3, activation='relu')(input_shape)
        lane_2 = Conv1D(filters=filters, kernel_size=4, activation='relu')(input_shape)
        lane_3= Conv1D(filters=filters, kernel_size=5, activation='relu')(input_shape)
    elif e_type == "ConceptNet":
        # Potentially integrate tokenizer into model if we switch to modifying fit functin's class weight instead of SMOTE

        # input_shape = Input(shape=(1,), dtype=tf.string)
        # vectorize_layer = TextVectorization(standardize="lower_and_strip_punctuation", split="whitespace",
        #
        input_shape = Input(shape=(max_length,))
        e = Embedding(vocabSize[con_index], 300, embeddings_initializer=Constant(embedding_matrix[con_index]), input_length=max_length,
                      trainable=False)(input_shape)
        lane_1 = Conv1D(filters=filters, kernel_size=3, activation='relu')(e)
        lane_2 = Conv1D(filters=filters, kernel_size=4, activation='relu')(e)
        lane_3 = Conv1D(filters=filters, kernel_size=5, activation='relu')(e)

    lane_1 = MaxPooling1D(pool_size=2)(lane_1)
    lane_2 = MaxPooling1D(pool_size=2)(lane_2)
    lane_3 = MaxPooling1D(pool_size=2)(lane_3)

    if know_infus_bool == True:
        isa_input = Input(shape=(max_length,))
        aff_input = Input(shape=(max_length,))
        isa = Embedding(vocabSize[isa_index], 100, embeddings_initializer=Constant(embedding_matrix[isa_index]),
                        input_length=max_length, trainable=False)(isa_input)
        aff = Embedding(vocabSize[aff_index], 100, embeddings_initializer=Constant(embedding_matrix[aff_index]),
                        input_length=max_length, trainable=False)(aff_input)

        isa_lane_1 = Conv1D(filters=filters, kernel_size=3, activation='relu')(isa)
        isa_lane_2 = Conv1D(filters=filters, kernel_size=4, activation='relu')(isa)
        isa_lane_3 = Conv1D(filters=filters, kernel_size=5, activation='relu')(isa)
        isa_lane_1 = MaxPooling1D(pool_size=2)(isa_lane_1)
        isa_lane_2 = MaxPooling1D(pool_size=2)(isa_lane_2)
        isa_lane_3 = MaxPooling1D(pool_size=2)(isa_lane_3)

        aff_lane_1 = Conv1D(filters=filters, kernel_size=3, activation='relu')(aff)
        aff_lane_2 = Conv1D(filters=filters, kernel_size=4, activation='relu')(aff)
        aff_lane_3 = Conv1D(filters=filters, kernel_size=5, activation='relu')(aff)
        aff_lane_1 = MaxPooling1D(pool_size=2)(aff_lane_1)
        aff_lane_2 = MaxPooling1D(pool_size=2)(aff_lane_2)
        aff_lane_3 = MaxPooling1D(pool_size=2)(aff_lane_3)
        merged = Concatenate(axis=1)([lane_1, lane_2, lane_3, isa_lane_1, isa_lane_2, isa_lane_3,
                                      aff_lane_1, aff_lane_2, aff_lane_3])
    else:
        merged = Concatenate(axis=1)([lane_1, lane_2, lane_3])


    merged = Flatten()(merged)
    merged = Dropout(param_grid["dropout"])(merged)
    if num_label == 2:
        out = Dense(num_label, activation='sigmoid')(merged)
    else:
        out = Dense(num_label, activation='softmax')(merged)

    if know_infus_bool == True:
        cnn = Model([input_shape, isa_input, aff_input], out)
    else:
        cnn = Model(input_shape, out)
    cnn.compile(optimizer=Adam(learning_rate=param_grid["learning_rate"]),
              loss=loss,
              metrics=metrics)

    cnn.summary()
    if platform.system() != "Linux":
        dot_img_file = f'CNN_{e_type}.png'
        tf.keras.utils.plot_model(cnn, to_file=dot_img_file, show_shapes=False,show_layer_names=False)
    return cnn

#CNN architecture from Ayaan paper
def cnnModel2(num_channels, num_features, num_labels, e_type):
    """
    parameters
    ------------------
    Xtrain
        numpy array of training set
    ytrain
        numpy array of training labels
    num_labels
        number of labels --either 4 or 5--
    rnd_seed
        random seed for reproducibilty
    """
    epochs = 80
    batch_size = 32

    # Convolutional Neural Network

    cnn = Sequential()

    cnn_path = "CNN"

    filters = 3
    kernal = 2

    cnn.add(Input(shape=(num_channels, num_features)))
    cnn.add(Conv1D(filters=filters, kernel_size=kernal, activation='relu'))
    cnn.add(Dropout(0.25))
    cnn.add(Flatten())
    cnn.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    if num_labels == 2:
        cnn.add(Dense(num_labels, activation='sigmoid'))
        cnn.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    else:
        cnn.add(Dense(num_labels, activation='softmax'))
        cnn.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    cnn.summary()
    return cnn

def RNNModel(param_grid, num_channels, num_features, num_label, e_type, model_name, modelType, know_infus_bool, emb_dim, preTrainDim = 300,
                 max_length = 512, vocabSize = [], embedding_matrix =[]):

    metrics = [tf.keras.metrics.CategoricalAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')]
    if num_label == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'

    if e_type == "BERT" and know_infus_bool == True:
        isa_index, aff_index = 0, 1
    elif e_type == "ConceptNet" and know_infus_bool == True:
        con_index, isa_index, aff_index = 0, 1, 2
    elif e_type == "ConceptNet" and know_infus_bool == False:
        con_index = 0


    if e_type == "BERT":
        input_shape = Input(shape=(num_channels, num_features), name=f"{model_name}_Input")
        if modelType == "GRU":
            embRNN = Bidirectional(GRU(param_grid["rnn_nodes"], return_sequences=True, dropout=0, recurrent_dropout=0), name=f"{model_name}_bi-{modelType}")(input_shape)
        elif modelType == "LSTM":
            embRNN = Bidirectional(LSTM(param_grid["rnn_nodes"], return_sequences=True, dropout=0.25, recurrent_dropout=0.25,
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                                       recurrent_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=seed)), name=f"{model_name}_bi-{modelType}")(input_shape)

    embAtt = Attention(dropout=0.25, name=f"{model_name}_Self_Attention")([embRNN, embRNN])

    if know_infus_bool == True:
        isa_input = Input(shape=(max_length,), name="IsaCore_Input")
        aff_input = Input(shape=(max_length,), name="AffectiveSpace_Input")

        isa = Embedding(vocabSize[isa_index], 100, embeddings_initializer=Constant(embedding_matrix[isa_index]),
                        input_length=max_length, trainable=False, name="IsaCore_Embeddings")(isa_input)
        aff = Embedding(vocabSize[aff_index], 100, embeddings_initializer=Constant(embedding_matrix[aff_index]),
                        input_length=max_length,trainable=False, name="AffectiveSpace_Embeddings")(aff_input)

        if modelType == "GRU":
            isaRNN = Bidirectional(GRU(param_grid["rnn_nodes"], return_sequences=True, dropout=0.25, recurrent_dropout=0.25), name=f"IsaCore_bi-{modelType}")(isa)
            affRNN = Bidirectional(GRU(param_grid["rnn_nodes"], return_sequences=True, dropout=0.25, recurrent_dropout=0.25), name=f"AffectiveSpace_bi-{modelType}")(aff)
        elif modelType == "LSTM":
            isaRNN = Bidirectional(LSTM(param_grid["rnn_nodes"], return_sequences=True, dropout=0.25, recurrent_dropout=0.25), name=f"IsaCore_bi-{modelType}")(isa)
            affRNN = Bidirectional(LSTM(param_grid["rnn_nodes"], return_sequences=True, dropout=0.25, recurrent_dropout=0.25), name=f"AffectiveSpace_bi-{modelType}")(aff)

        isaAtt = Attention(dropout=0.25, name="IsaCore_Self_Attention")([isaRNN, isaRNN])
        affAtt = Attention(dropout=0.25, name="AffectiveSpace_Self_Attention")([affRNN, affRNN])

        merged = Concatenate(axis=-1, name="Concatenation")([embAtt, isaAtt, affAtt])
        flattened = Flatten(name="Flattening")(merged)
    else:
        flattened = Flatten(name="Flattening")(embAtt)

    # dense1 = Dense(param_grid["1st_dense"], activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=seed),
    #                name="1st-fully-connected")(flattened)
    # drop1 = Dropout(param_grid["dropout"], name="1st-dropout", seed=seed)(dense1)
    # dense2 = Dense(param_grid["2nd_dense"], activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=seed),
    #                name="2nd-fully-connected")(drop1)
    # drop2 = Dropout(param_grid["dropout"], name="2nd-dropout", seed=seed)(dense2)
    # out = Dense(num_label, activation='softmax', name="Output")(drop2)

    dense1 = Dense(param_grid["1st_dense"], activation='relu',name="1st-fully-connected")(flattened)
    drop1 = Dropout(param_grid["dropout"], name="1st-dropout")(dense1)
    dense2 = Dense(param_grid["2nd_dense"], activation='relu',name="2nd-fully-connected")(drop1)
    drop2 = Dropout(param_grid["dropout"], name="2nd-dropout")(dense2)
    out = Dense(num_label, activation='softmax', name="Output")(drop2)

    if know_infus_bool == True:
        nMod = Model([input_shape, isa_input, aff_input], out)
    else:
        nMod = Model(input_shape, out)

    nMod.compile(optimizer=Adam(learning_rate=param_grid["learning_rate"]),
                   loss=loss,
                   metrics=metrics)

    nMod.summary()
    if platform.system() != "Linux":
        dot_img_file = f'Cascade_{e_type}.png'
        tf.keras.utils.plot_model(nMod, to_file=dot_img_file, show_shapes=False, show_layer_names=False)
    return nMod

def RNNModel2(param_grid, num_channels, num_features, num_label, e_type, modelType, know_infus_bool, emb_dim, preTrainDim = 300,
                 max_length = 512, vocabSize = [], embedding_matrix =[]):

    metrics = [tf.keras.metrics.CategoricalAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')]
    if num_label == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'

    if e_type == "BERT" and know_infus_bool == True:
        isa_index, aff_index = 0, 1
    elif e_type == "ConceptNet" and know_infus_bool == True:
        con_index, isa_index, aff_index = 0, 1, 2
    elif e_type == "ConceptNet" and know_infus_bool == False:
        con_index = 0


    if e_type == "BERT":
        input_shape = Input(shape=(num_channels, num_features), name="BERT_Input")
        if modelType == "GRU":
            # embRNN = GRU(param_grid["rnn_nodes"], return_sequences=True, dropout=0.25, recurrent_dropout=0.25,
            #                            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
            #                            recurrent_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=seed), name=f"BERT_{modelType}")(input_shape)

            # embRNN = GRU(param_grid["rnn_nodes"], return_sequences=True, dropout=0, recurrent_dropout=0,
            #              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
            #              recurrent_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=seed),
            #              name=f"BERT_{modelType}")(input_shape)

            embRNN = GRU(param_grid["rnn_nodes"], return_sequences=True, dropout=0.25, recurrent_dropout=0.25,name=f"BERT_{modelType}")(input_shape)

            # embRNN = GRU(param_grid["rnn_nodes"], return_sequences=True, dropout=0, recurrent_dropout=0,name=f"BERT_{modelType}")(input_shape)

        elif modelType == "LSTM":
            embRNN = LSTM(param_grid["rnn_nodes"], return_sequences=True, dropout=0.25, recurrent_dropout=0.25,
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                                       recurrent_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=seed), name=f"BERT_{modelType}")(input_shape)
    elif e_type == "ConceptNet":
        input_shape = Input(shape=(max_length,), name="ConceptNet_Input")
        emb = Embedding(vocabSize[con_index], preTrainDim, embeddings_initializer=Constant(embedding_matrix[con_index]),
                        input_length=max_length, trainable=False, name="ConceptNet_Embeddings")(input_shape)

        if modelType == "GRU":
            embRNN = GRU(param_grid["rnn_nodes"], return_sequences=True, dropout=0.25, recurrent_dropout=0.25,
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                                       recurrent_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=seed), name=f"ConceptNet_{modelType}")(emb)
        elif modelType == "LSTM":
            embRNN = LSTM(param_grid["rnn_nodes"], return_sequences=True, dropout=0.25, recurrent_dropout=0.25,
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed),
                                       recurrent_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=seed), name=f"ConceptNet_{modelType}")(emb)

    embAtt = Attention(dropout=0.25, name=f"{e_type}_Self_Attention")([embRNN, embRNN])

    if know_infus_bool == True:
        isa_input = Input(shape=(max_length,), name="IsaCore_Input")
        aff_input = Input(shape=(max_length,), name="AffectiveSpace_Input")

        isa = Embedding(vocabSize[isa_index], 100, embeddings_initializer=Constant(embedding_matrix[isa_index]),
                        input_length=max_length, trainable=False, name="IsaCore_Embeddings")(isa_input)
        aff = Embedding(vocabSize[aff_index], 100, embeddings_initializer=Constant(embedding_matrix[aff_index]),
                        input_length=max_length,trainable=False, name="AffectiveSpace_Embeddings")(aff_input)

        if modelType == "GRU":
            isaRNN = Bidirectional(GRU(param_grid["rnn_nodes"], return_sequences=True, dropout=0.25, recurrent_dropout=0.25), name=f"IsaCore_bi-{modelType}")(isa)
            affRNN = Bidirectional(GRU(param_grid["rnn_nodes"], return_sequences=True, dropout=0.25, recurrent_dropout=0.25), name=f"AffectiveSpace_bi-{modelType}")(aff)
        elif modelType == "LSTM":
            isaRNN = Bidirectional(LSTM(param_grid["rnn_nodes"], return_sequences=True, dropout=0.25, recurrent_dropout=0.25), name=f"IsaCore_bi-{modelType}")(isa)
            affRNN = Bidirectional(LSTM(param_grid["rnn_nodes"], return_sequences=True, dropout=0.25, recurrent_dropout=0.25), name=f"AffectiveSpace_bi-{modelType}")(aff)

        isaAtt = Attention(dropout=0.25, name="IsaCore_Self_Attention")([isaRNN, isaRNN])
        affAtt = Attention(dropout=0.25, name="AffectiveSpace_Self_Attention")([affRNN, affRNN])

        merged = Concatenate(axis=-1, name="Concatenation")([embAtt, isaAtt, affAtt])
        flattened = Flatten(name="Flattening")(merged)
    else:
        flattened = Flatten(name="Flattening")(embAtt)

    # dense1 = Dense(param_grid["1st_dense"], activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=seed),
    #                name="1st-fully-connected")(flattened)
    # drop1 = Dropout(param_grid["dropout"], name="1st-dropout", seed=seed)(dense1)
    # dense2 = Dense(param_grid["2nd_dense"], activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=seed),
    #                name="2nd-fully-connected")(drop1)
    # drop2 = Dropout(param_grid["dropout"], name="2nd-dropout", seed=seed)(dense2)

    dense1 = Dense(param_grid["1st_dense"], activation='relu',name="1st-fully-connected")(flattened)
    drop1 = Dropout(param_grid["dropout"], name="1st-dropout")(dense1)
    dense2 = Dense(param_grid["2nd_dense"], activation='relu',name="2nd-fully-connected")(drop1)
    drop2 = Dropout(param_grid["dropout"], name="2nd-dropout")(dense2)

    out = Dense(num_label, activation='softmax', name="Output")(drop2)

    if know_infus_bool == True:
        nMod = Model([input_shape, isa_input, aff_input], out)
    else:
        nMod = Model(input_shape, out)

    nMod.compile(optimizer=Adam(learning_rate=param_grid["learning_rate"]),
                   loss=loss,
                   metrics=metrics)

    nMod.summary()
    if platform.system() != "Linux":
        dot_img_file = f'Cascade_{e_type}.png'
        tf.keras.utils.plot_model(nMod, to_file=dot_img_file, show_shapes=False, show_layer_names=False)
    return nMod

def multiTaskModel(transformerModel, maxLength, CSSRS_n_label, UMD_n_label):
    # CSSRS_Input = Input(shape=(maxLength,), name="CSSRS_Input", dtype='int32')
    # UMD_Input = Input(shape=(maxLength,), name="UMD_Input", dtype='int32')
    #
    CSSRS_Input_Ids = Input(shape=(maxLength,), name="CSSRS_Input_Ids", dtype='int32')
    CSSRS_Attention_Mask = Input(shape=(maxLength,), name="CSSRS_Attention_Mask", dtype='int32')
    CSSRS_Token_Types = Input(shape=(maxLength,), name="CSSRS_Token_Types", dtype='int32')
    UMD_Input_Ids = Input(shape=(maxLength,), name="UMD_Input_Ids", dtype='int32')
    UMD_Attention_Mask = Input(shape=(maxLength,), name="UMD_Attention_Mask", dtype='int32')
    UMD_Token_Types = Input(shape=(maxLength,), name="UMD_Token_Types", dtype='int32')

    # output_embed1 = transformerModel(CSSRS_Input)
    # output_embed2 = transformerModel(UMD_Input)

    output_embed1 = transformerModel([CSSRS_Input_Ids, CSSRS_Attention_Mask, CSSRS_Token_Types])
    output_embed2 = transformerModel([UMD_Input_Ids, UMD_Attention_Mask, UMD_Token_Types])

    # output_embed1 = transformerModel([CSSRS_Input["input_ids"], CSSRS_Input["attention_mask"]])
    # output_embed2 = transformerModel([UMD_Input["input_ids"], UMD_Input["attention_mask"]])
    # output_embed1 = transformerModel([CSSRS_Input_Ids, CSSRS_Attention_Mask])
    # output_embed2 = transformerModel([UMD_Input_Ids, UMD_Attention_Mask])

    dropout1 = Dropout(0.1)(output_embed1[1])
    dropout2 = Dropout(0.1)(output_embed2[1])

    CSSRS_out = Dense(CSSRS_n_label, activation="softmax", name="CSSRS_Output")(dropout1)
    UMD_out = Dense(UMD_n_label, activation="softmax", name="UMD_Output")(dropout2)

    # return Model([CSSRS_Input_Ids, UMD_Input_Ids], [CSSRS_out, UMD_out])
    # return Model([CSSRS_Input, UMD_Input], [CSSRS_out, UMD_out])
    # return Model([CSSRS_Input_Ids, CSSRS_Attention_Mask], [CSSRS_out])
    return Model([CSSRS_Attention_Mask, CSSRS_Input_Ids, CSSRS_Token_Types, UMD_Attention_Mask, UMD_Input_Ids, UMD_Token_Types], [CSSRS_out, UMD_out])


def multiTaskModel1(transformerModel, datasetName, maxLength, n_label):


    Input_Ids = Input(shape=(maxLength,), name=f"input_ids", dtype='int32')
    Attention_Mask = Input(shape=(maxLength,), name=f"attention_mask", dtype='int32')
    Token_Types = Input(shape=(maxLength,), name=f"token_type_ids", dtype='int32')

    output_embed1 = transformerModel([Input_Ids, Attention_Mask, Token_Types])

    dropout1 = Dropout(0.1)(output_embed1[1])

    out = Dense(n_label, activation="softmax", name=f"{datasetName}_Output")(dropout1)
    return Model([Input_Ids, Token_Types, Attention_Mask], out)

