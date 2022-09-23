#!/usr/bin/env python3

from Libraries import *
from modelFunctions import denseNN, cnnModel, cnnModel2, objectiveFunctionCNN, RNNModel, objectiveFunctionRNN
from HelperFunctions import getEmbeddings,bert_encode, getTokens, getLabels, extractList, \
    getXfromBestModelfromTrials, getStatistics, printPredictions, printOverallResults, onehotEncode, getSummStats

def runFold(outputPath, filespath, modelType, know_infus_bool, emb_type, max_length, num_labels, embed_dimen,split_Bool, CV_Bool, param_tune,
            smoteBool, weightBool, hyperparameters, number_of_folds, fold_num, X_train_fold, y_train_fold, X_test_fold, y_test_fold, embToken,
            isacore_Embeddings = {}, affectiveSpace_Embeddings = {}):

    if emb_type == "BERT":
        tokenizer = embToken[0]
        model = embToken[1]
    elif emb_type == "ConceptNet":
        concept_Embeddings = embToken

    if know_infus_bool == True:
        isa = [x for x in isacore_Embeddings.keys() if x != ""]
        aff = [x for x in affectiveSpace_Embeddings.keys() if x != ""]
        # list_of_vocab = list(set(con).union(set(isa), set(aff)))

        isa_vocab_size, isa_word_index, \
        isa_train_tokens, isa_test_tokens= getTokens(standardization = "lower_and_strip_punctuation",
                                                     splitMethod="whitespace", maxTokens = None,
                                                     outputLength = max_length, vocab = isa, num_ngrams = 3,
                                                     Xtrain=X_train_fold, Xtest=X_test_fold)

        aff_vocab_size, aff_word_index, \
        aff_train_tokens, aff_test_tokens = getTokens(standardization="lower_and_strip_punctuation",
                                                      splitMethod="whitespace", maxTokens=None,
                                                      outputLength=max_length, vocab=aff, num_ngrams=3,
                                                      Xtrain=X_train_fold, Xtest=X_test_fold)

    if emb_type == "BERT":
        train_input_ids, train_attention_masks = bert_encode(X_train_fold, tokenizer, max_length)
        test_input_ids, test_attention_masks = bert_encode(X_test_fold, tokenizer, max_length)

    elif emb_type == "ConceptNet":
        con = [x for x in concept_Embeddings.keys() if x != ""]
        con_vocab_size, con_word_index, \
        con_train_tokens, con_test_tokens = getTokens(standardization="lower_and_strip_punctuation",
                                                      splitMethod="whitespace", maxTokens=None,
                                                      outputLength=max_length, vocab=con, num_ngrams=3,
                                                      Xtrain=X_train_fold, Xtest=X_test_fold)

        # Get embedding matrix
        concept_embedding_matrix = np.zeros((con_vocab_size, preTrainDim))
        j = 0
        for word, i in con_word_index.items():
            embedding_vector = concept_Embeddings.get(word)
            if embedding_vector is not None:
                concept_embedding_matrix[i] = embedding_vector
            else:
                j = j + 1
        # print(f"Out of words count for Conceptnet: {j}")
        print(f"Total Vocab Size of ConceptNet is {con_vocab_size}")


    if know_infus_bool == True:
        # Get Isacore embedding matrix
        isa_embedding_matrix = np.zeros((isa_vocab_size, 100))
        j = 0
        for word, i in isa_word_index.items():
            embedding_vector = isacore_Embeddings.get(word)
            if embedding_vector is not None:
                isa_embedding_matrix[i] = embedding_vector
            else:
                j = j + 1
        # print(f"Out of words count for isaCore: {j}")
        # Get AffectiveSpace embedding matrix
        affectiveSpace_embedding_matrix = np.zeros((aff_vocab_size, 100))
        j = 0
        for word, i in aff_word_index.items():
            embedding_vector = affectiveSpace_Embeddings.get(word)
            if embedding_vector is not None:
                affectiveSpace_embedding_matrix[i] = embedding_vector
            else:
                j = j + 1
        # print(f"Out of words count for affectiveSpace: {j}")

        print(f"Total Vocab Size of isaCore is {isa_vocab_size}")
        print(f"Total Vocab Size of affectiveSpace is {aff_vocab_size}")

    # Applying SMOTE to training set
    # Random seed set for now to make sure all pipelines overfit the same way
    if smoteBool == True:
        valueCounts = pd.Series(y_train_fold).value_counts()
        print("Before:", valueCounts)
        # y_train_fold.value_counts()
        over = RandomOverSampler(random_state=SMOTE_random_seed)
        steps = [('o', over)]
        pipeline = Pipeline(steps=steps)
        if emb_type == "BERT":
            train_input_ids, _ = pipeline.fit_resample(train_input_ids, y_train_fold)
            if not know_infus_bool:
                train_attention_masks, y_train_fold = pipeline.fit_resample(train_attention_masks, y_train_fold)
            else:
                train_attention_masks, _ = pipeline.fit_resample(train_attention_masks, y_train_fold)
                isa_train_tokens, _ = pipeline.fit_resample(isa_train_tokens, y_train_fold)
                aff_train_tokens, y_train_fold = pipeline.fit_resample(aff_train_tokens, y_train_fold)

        if emb_type == "ConceptNet":
            if not know_infus_bool:
                con_train_tokens, y_train_fold = pipeline.fit_resample(con_train_tokens, y_train_fold)
            else:
                con_train_tokens, _ = pipeline.fit_resample(con_train_tokens, y_train_fold)
                isa_train_tokens, _ = pipeline.fit_resample(isa_train_tokens, y_train_fold)
                aff_train_tokens, y_train_fold = pipeline.fit_resample(aff_train_tokens, y_train_fold)
            print("After:", pd.Series(y_train_fold).value_counts())
    # Convert to BERT embeddings for BERT models
    # Rename tokens for ConceptNet models for simplicity of coding
    if emb_type == "BERT":
        # text, tokenizer, model, 2d or 3d embeddings
        X_train_emb_fold = getEmbeddings(train_input_ids, train_attention_masks, model, embed_dimen)
        X_test_emb_fold = getEmbeddings(test_input_ids, test_attention_masks, model, embed_dimen)
        number_channels, number_features = X_train_emb_fold.shape[1], X_train_emb_fold.shape[2]
    elif emb_type == "ConceptNet":
        X_train_emb_fold = con_train_tokens
        X_test_emb_fold = con_test_tokens
        number_channels, number_features = 512, 768

    X_train_emb_fold = tf.convert_to_tensor(X_train_emb_fold)
    X_test_emb_fold = tf.convert_to_tensor(X_test_emb_fold)

    if know_infus_bool == True:
        isa_train_tokens = tf.convert_to_tensor(isa_train_tokens)
        isa_test_tokens = tf.convert_to_tensor(isa_test_tokens)
        aff_train_tokens = tf.convert_to_tensor(aff_train_tokens)
        aff_test_tokens = tf.convert_to_tensor(aff_test_tokens)

    if know_infus_bool == True:
        modelTrain = [X_train_emb_fold, isa_train_tokens, aff_train_tokens]
        modelTest = [X_test_emb_fold, isa_test_tokens, aff_test_tokens]
    else:
        modelTrain = X_train_emb_fold
        modelTest = X_test_emb_fold


    if know_infus_bool == True and emb_type == "ConceptNet":
        vocab_sizes = [con_vocab_size, isa_vocab_size, aff_vocab_size]
        embed_matrices = [concept_embedding_matrix, isa_embedding_matrix, affectiveSpace_embedding_matrix]
    elif know_infus_bool == False and emb_type == "ConceptNet":
        vocab_sizes = [con_vocab_size]
        embed_matrices = [concept_embedding_matrix]
    elif know_infus_bool == True and emb_type == "BERT":
        vocab_sizes = [isa_vocab_size, aff_vocab_size]
        embed_matrices = [isa_embedding_matrix, affectiveSpace_embedding_matrix]
    elif know_infus_bool == False and emb_type == "BERT":
        vocab_sizes = []
        embed_matrices = []

    if weightBool == True:
        # Generate class weights & One-hot encode labels
        print("\nClass weight")
        num_classes = len(pd.Series(y_train_fold.numpy()).unique())
        onehot = pd.get_dummies(pd.Series(y_train_fold.numpy()), drop_first=False)
        class_counts = onehot.sum(axis=0).values
        total_count = sum(class_counts)
        class_rate = [(total_count / (num_classes * x)) for x in class_counts]
        class_weights = dict(enumerate(class_rate))
        print("num_classes: ", num_classes, "class_counts: ", class_counts, "total_count: ", total_count,
              "class_weights: ", class_weights)

        y_train_fold = convert_to_tensor(onehot)
        onehotTest = pd.get_dummies(pd.Series(y_test_fold.numpy()), drop_first=False)
        y_test_fold = convert_to_tensor(onehotTest)
    else:
        onehot = pd.get_dummies(pd.Series(y_train_fold), drop_first=False)
        y_train_fold = convert_to_tensor(onehot)
        onehotTest = pd.get_dummies(pd.Series(y_test_fold.numpy()), drop_first=False)
        y_test_fold = convert_to_tensor(onehotTest)


    if know_infus_bool == True:
        checkpointName = f"{emb_type}_{modelType}_with_KI_best_model.h5"
    else:
        checkpointName = f"{emb_type}_{modelType}_no_KI_best_model.h5"

    es = EarlyStopping(monitor='val_auc', mode="max", patience=10, min_delta=0)
    mc = ModelCheckpoint(checkpointName, monitor='val_auc', mode='max', verbose=0,
                         save_best_only=True)

    if param_tune == True:
        param_grid = hyperparameters
        tpe_trials = Trials()
        tpe_best = []

        if modelType == "CNN":
            objectiveFunc = partial(objectiveFunctionCNN, num_channels=number_channels,num_features=number_features,
                                    Xtrain=modelTrain, ytrain=y_train_fold, Xtest=modelTest, ytest=y_test_fold,
                                    num_label=num_labels, modelType = modelType, e_type=emb_type, max_length=max_length,
                                    know_infus_bool=know_infus_bool, vocabSize=vocab_sizes, embedding_matrix=embed_matrices,
                                    es=es, mc=mc)
        elif modelType == "GRU" or modelType == "LSTM":
            objectiveFunc = partial(objectiveFunctionRNN, num_channels=number_channels, num_features=number_features,emb_dim=embed_dimen,
                                    Xtrain=modelTrain, ytrain=y_train_fold, Xtest=modelTest, ytest=y_test_fold,
                                    n_lab=num_labels, e_type=emb_type, modelType = modelType, max_length=max_length,
                                    know_infus_bool=know_infus_bool, vocabSize=vocab_sizes, embedding_matrix=embed_matrices,
                                    es=es, mc=mc)

        tpe_best = fmin(fn=objectiveFunc, space=param_grid, algo=tpe.suggest,
                        max_evals=global_max_evals, trials=tpe_trials)
        hyperparameters = space_eval(param_grid, tpe_best)
        print("Best: ", getXfromBestModelfromTrials(tpe_trials, 'loss'), hyperparameters)

    print("--------------------")
    print(hyperparameters)
    print("--------------------")

    if modelType == "CNN":
        nnModel = cnnModel(hyperparameters, number_channels, number_features, num_labels, emb_type, know_infus_bool,
                           max_length, vocab_sizes, embed_matrices)
    elif modelType == "GRU" or modelType == "LSTM":
        nnModel = RNNModel(hyperparameters, number_channels, number_features, num_labels, emb_type, modelType, know_infus_bool,
                           embed_dimen, preTrainDim, max_length, vocab_sizes, embed_matrices)

    if weightBool == True:
        history = nnModel.fit(modelTrain, y_train_fold,
                              validation_data=(modelTrain, y_train_fold),
                              epochs=hyperparameters["epochs"],
                              batch_size=hyperparameters["batch_size"], callbacks=[es, mc],
                              class_weight=class_weights,verbose=2)
    else:

        history = nnModel.fit(modelTrain, y_train_fold,
                              validation_data=(modelTrain, y_train_fold),
                              epochs=hyperparameters["epochs"],
                              batch_size=hyperparameters["batch_size"], callbacks=[es, mc],
                              verbose=2)

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    nnModel = load_model(checkpointName)
    scores = nnModel.evaluate(modelTest, y_test_fold, verbose=0)
    y_pred_proba = nnModel.predict(modelTest)


    return nnModel, history, scores, y_pred_proba, hyperparameters

def runModel(outputPath, filespath, modelType, know_infus_bool, emb_type, max_length, num_labels, embed_dimen,split_Bool, CV_Bool, param_tune,
             smoteBool, weightBool, hyperparameters, number_of_folds):

    if tf.test.gpu_device_name():
        print('GPU: {}'.format(tf.test.gpu_device_name()))
    else:
        print('CPU version')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    startTime = datetime.now()
    CSSRS = pd.read_csv(filespath)
    # getSummStats(CSSRS)
    # CSSRS.rename(columns={"selftext":"Post", "is_suicide":"Label"}, inplace=True)

    # Get copy of original dataset and append embeddings #W
    df = CSSRS.copy(deep=True)
    if num_labels == 4 or num_labels == 5:
        df, inv_map = getLabels(df, num_labels)
    elif num_labels == 2:
        df = df.sample(frac=0.5, random_state=shuffle_rnd_state)

    ## Extract List from string then join back together (fixes string)
    extractList(df)
    df["Post"] = df["Post"].apply(lambda x: " ".join(x))
    # text = df["Post"].apply(lambda x: x.strip('][').replace("\', '", " ").lower())
    text = df["Post"].apply(lambda x: x.lower())
    labels = df["Label"]

    test_size=0.25
    if split_Bool == True:
        X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size = test_size, shuffle = True, stratify=labels, random_state=split_random_seed)
        new_df = pd.DataFrame({"Post": list(X_train), "Label": y_train}, columns=['Post', 'Label'])
    else:
        new_df = pd.DataFrame({"Post": text, "Label": labels}, columns=['Post', 'Label'])

    if emb_type == "BERT":
        #Import bert model
        model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
    elif emb_type == "ConceptNet":
        concept_Embeddings = dict()
        f = open(numberbatch_path, encoding="utf8")
        for line in f:
            values = line.split()
            word = str(values[0])
            coefs = np.asarray(values[1:], dtype='float32')
            concept_Embeddings[word] = coefs
        f.close()
        concept_Embeddings = {k.replace("_", " "): v for k, v in concept_Embeddings.items()}
        concept_Embeddings = {k.translate(str.maketrans('', '', string.punctuation)).strip(): v for k, v in
                              concept_Embeddings.items()}

    if know_infus_bool == True:
        df = pd.read_csv(isacore_path, header=None, index_col=0)
        df = df.drop(index="#NAME?")
        isacore_Embeddings = dict()
        for index, row in df.iterrows():
            isacore_Embeddings[str(index)] = row.to_numpy()
        isacore_Embeddings = {k.translate(str.maketrans('', '', string.punctuation)): v for k, v in
                              isacore_Embeddings.items()}

        df = pd.read_csv(affectiveSpace_path, header=None)
        df[0] = df[0].str.replace("_", " ")
        df.set_index(0, inplace=True)
        affectiveSpace_Embeddings = dict()
        for index, row in df.iterrows():
            affectiveSpace_Embeddings[str(index)] = row.to_numpy()

        # concept_vocab_size = len(list(embeddings_index.keys()))
        # isacore_vocab_size = len(list(isacore_Embeddings.keys()))
        # affectiveSpace_vocab_size = len(list(isacore_Embeddings.keys()))
        #
        # print(f"Total Vocab Size of ConceptNet is {concept_vocab_size}")
        # print(f"Total Vocab Size of isaCore is {isacore_vocab_size}")
        # print(f"Total Vocab Size of affectiveSpace is {affectiveSpace_vocab_size}")

    new_df = new_df.sample(frac=1, random_state=KFold_shuffle_random_seed)

    if emb_type == "BERT":
        embToken = [tokenizer, model]
    elif emb_type == "ConceptNet":
        embToken = concept_Embeddings

    if CV_Bool == True:
        # Define per-fold score containers
        acc_per_fold = []
        loss_per_fold = []
        fold_stats = []
        fold_matrix = []
        fold_hyper = []
        test_size_per_fold = []

        whole_results = pd.DataFrame({"Actual":pd.Series(dtype=int), "Predicted":pd.Series(dtype=int),
                                      "PredictedProba":pd.Series(dtype=int), "Fold":pd.Series(dtype=int)})
        fold_results = []
        sfk = StratifiedKFold(n_splits=number_of_folds, shuffle=False)
        fold_num = 1
        for train_indx, test_indx in sfk.split(new_df["Post"], new_df["Label"]):

            fold_train = new_df.iloc[train_indx].copy()
            X_train_fold = fold_train["Post"]
            y_train_fold = fold_train["Label"]

            fold_test = new_df.iloc[test_indx].copy()
            X_test_fold = fold_test["Post"]
            y_test_fold = fold_test["Label"]

            y_train_fold = tf.convert_to_tensor(y_train_fold)
            y_test_fold = tf.convert_to_tensor(y_test_fold)
            # Generate a print
            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_num} ...')


            if know_infus_bool == True:
                nnModel, history, scores, y_pred_proba, hyperparameters = runFold(outputPath, filespath, modelType, know_infus_bool, emb_type, max_length,
                                                                                  num_labels, embed_dimen,split_Bool, CV_Bool, param_tune,
                                                                                  smoteBool, weightBool, hyperparameters, number_of_folds, fold_num,
                                                                                  X_train_fold, y_train_fold, X_test_fold, y_test_fold,
                                                                                  embToken, isacore_Embeddings, affectiveSpace_Embeddings)
            else:
                nnModel, history, scores, y_pred_proba, hyperparameters = runFold(outputPath, filespath, modelType, know_infus_bool,
                                                                                  emb_type, max_length,
                                                                                  num_labels, embed_dimen, split_Bool,
                                                                                  CV_Bool, param_tune,
                                                                                  smoteBool, weightBool, hyperparameters,
                                                                                  number_of_folds, fold_num,
                                                                                  X_train_fold, y_train_fold,
                                                                                  X_test_fold, y_test_fold,
                                                                                  embToken)


            train_auc = history.history['auc']
            val_auc = history.history['val_auc']
            train_acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
            epochs = range(len(train_auc))


            fold_results.append({"train_auc":train_auc, "val_auc":val_auc, "train_acc":train_acc, "val_acc":val_acc,
                            "train_loss":train_loss, "val_loss":val_loss, "epochs":epochs})

            # Generate generalization metrics
            print(f'Score for fold {fold_num}: {nnModel.metrics_names[0]} of {scores[0]}; {nnModel.metrics_names[1]} of {scores[1] * 100}%')
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])
            fold_hyper.append(hyperparameters)

            y_pred = np.argmax(y_pred_proba,axis=1)
            whole_results = pd.concat([whole_results, pd.DataFrame({"Actual":y_test_fold.numpy().tolist(), "Predicted":y_pred.tolist(),
                                                                    "PredictedProba":y_pred_proba.tolist(), "Fold":fold_num})], ignore_index=True)

            print(classification_report(y_test_fold, y_pred))

            #contains precision, recall, and f1 score for each class
            report = classification_report(y_test_fold, y_pred, output_dict=True)

            #Get only precision, recall, f1-score, and support statistics
            # filtered_report = {str(label): report[str(label)] for label in range(num_labels)}

            matrix = confusion_matrix(y_test_fold, y_pred)
            print(f"{num_labels}-label confusion matrix")
            print(matrix)
            #Increase Fold Number
            fold_num = fold_num + 1

            tf.keras.backend.clear_session()
            tf.random.set_seed(seed)
        # == Provide average scores ==
        print('------------------------------------------------------------------------')
        print('Score per fold')
        for i in range(0, len(acc_per_fold)):
            print('------------------------------------------------------------------------')
            print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')

        overallResults = getStatistics(outputPath, whole_results["Actual"], whole_results["PredictedProba"], whole_results["Predicted"], num_labels)
        # whole_results.to_csv(os.path.join(outputPath, "Actual_vs_Predicted.csv"), index=False)
        endTime = datetime.now()
        elapsedTime = endTime - startTime

        printOverallResults(outputPath, f"New OverallResults {num_labels}Label.csv", num_labels,emb_type,max_length, smoteBool, split_Bool,
                            number_of_folds, modelType, know_infus_bool, param_tune, overallResults, fold_hyper, elapsedTime, whole_results, fold_results)

    # No CV Folds
    else:
        y_train = tf.convert_to_tensor(y_train)
        y_test = tf.convert_to_tensor(y_test)

        if know_infus_bool == True:
            nnModel, history, scores, y_pred_proba, hyperparameters = runFold(outputPath, filespath, modelType, emb_type,
                                                                              max_length, num_labels, embed_dimen, split_Bool, CV_Bool,
                                                                              param_tune, smoteBool, hyperparameters, number_of_folds,
                                                                              X_train, y_train, X_test, y_test, embToken,
                                                                              isacore_Embeddings, affectiveSpace_Embeddings)
        else:
            nnModel, history, scores, y_pred_proba, hyperparameters = runFold(outputPath, filespath, modelType, emb_type,
                                                                              max_length, num_labels, embed_dimen,
                                                                              split_Bool, CV_Bool, param_tune, smoteBool,
                                                                              hyperparameters, number_of_folds,
                                                                              X_train, y_train, X_test, y_test, embToken)
        y_pred = np.argmax(y_pred_proba, axis=1)

        whole_results = pd.DataFrame({"Actual": y_test.numpy().tolist(), "Predicted": y_pred.tolist(),
                                          "PredictedProba": y_pred_proba.tolist()})

        overallResults = getStatistics(outputPath, whole_results["Actual"], whole_results["PredictedProba"],
                                       whole_results["Predicted"], num_labels)
        fold_results = []
        # printPredictions(y_test, y_pred, num_labels, outputPath)
        endTime = datetime.now()
        elapsedTime = endTime - startTime

        printOverallResults(outputPath, f"OverallResults {num_labels}Label (no CV).csv", num_labels, emb_type, max_length, smoteBool,
                            split_Bool, number_of_folds, modelType, know_infus_bool, param_tune, overallResults, hyperparameters,
                            elapsedTime, whole_results, fold_results)


def main():
    if platform.system() == "Windows":
        filePath = r"D:\Summer 2022 Project\Reddit C-SSRS\500_Reddit_users_posts_labels.csv"
        outputPath = r"C:\Users\dmlee\Desktop\Summer_Project\Summer 2022\Output\CSSRS"
    elif platform.system() == "Linux":
        filePath = r"/ddn/home12/r3102/files/500_Reddit_users_posts_labels.csv"
        outputPath = r"/ddn/home12/r3102/results"

    embeddingType = "BERT"
    # embeddingType = "ConceptNet"

    # modtype = "CNN"
    modtype = "GRU"
    # modtype = "LSTM"

    maxLength = 512

    # knowledgeInfusion = True
    knowledgeInfusion = False

    # smote_bool = True
    smote_bool = False

    weight_bool = True
    # weight_bool = False

    num_labels = 4
    emb_dim = "3d"
    number_of_folds = 5
    split = False
    cross_validation = True
    parameter_tune = False
    if parameter_tune == True:
        param_grid = {"epochs": hp.choice("epochs", [10, 25, 50]),
                      "batch_size": hp.choice("batch_size", [4, 24, 32]),
                      "dropout": hp.choice("droupout", [0.1, 0.2, 0.3, 0.4, 0.5]),
                      "learning_rate":hp.choice("learning_rate", [0.01, 0.005, 0.001]),
                      "rnn_nodes": hp.choice("rnn_nodes", [128, 256])}
    else:                                        #Default Values
        param_grid = {"batch_size": 32,          #32
                      "dropout": 0.25,           #0.25 for RNN, 0.3 for CNN
                      "epochs": 10,              #10
                      "learning_rate":0.001,     #0.001
                      "rnn_nodes":128,           #128
                      "1st_dense":300,           #300
                      "2nd_dense":100}           #100

    if smote_bool == True and weight_bool == True:
        print("Both SMOTE and loss weighting set to True. For now, please only choose one. Closing...")
        exit()
    print("-----------------------------------")
    print(f"Embedding Type: {embeddingType}")
    print(f"Model Type: {modtype}")
    print(f"Sentence Length: {maxLength}")
    if smote_bool == True:
        print("With SMOTE")
    elif weight_bool == True:
        print("With weighted loss")
    print(f"{'No' if knowledgeInfusion == False else 'with'} Knowledge Infusion")
    print(f"{'No' if parameter_tune == False else 'with'} Parameter Tuning")
    print("-----------------------------------")
    runModel(outputPath, filePath, modtype, knowledgeInfusion, embeddingType, maxLength, num_labels, emb_dim, split, cross_validation,
             parameter_tune, smote_bool, weight_bool, param_grid, number_of_folds)

global_max_evals = 30
if platform.system() == "Windows":
    numberbatch_path = r"D:\Summer 2022 Project\numberbatch-en.txt"
    isacore_path = r"D:\Summer 2022 Project\isacore\isacore.csv"
    affectiveSpace_path = r"D:\Summer 2022 Project\affectivespace\affectivespace.csv"
elif platform.system() == "Linux":
    numberbatch_path = r"/ddn/home12/r3102/files/numberbatch-en.txt"
    isacore_path = r"/ddn/home12/r3102/files/isacore.csv"
    affectiveSpace_path = r"/ddn/home12/r3102/files/affectivespace.csv"

# glove_100d_path = r"D:\Summer 2022 Project\glove.6B.100d.txt"
preTrainDim = 300
seed = 99
main()
