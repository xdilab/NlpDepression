#!/usr/bin/env python3

from Libraries import *
from ModelFunctions import denseNN, cnnModel, cnnModel2, objectiveFunctionCNN, RNNModel, objectiveFunctionRNN
from HelperFunctions import getLabels, extractList,getXfromBestModelfromTrials, printPredictions, \
    printOverallResults, onehotEncode
from EmbeddingFunctions import BERT_embeddings, getTokens, customDataset, runModel, MaskingFunction, getFeatures
from StatisticsFunctions import getStatistics, getSummStats
from ImportFunctions import importUMD, importCSSRS, getModel, getTokenizer, getRegularModel

def runFold(outputPath, filespath, model, model_name, tokenizer, modelType, emb_type, max_length, num_labels, embed_dimen, boolDict,
            hyperparameters, n_folds, fold_num, X_train_fold, y_train_fold, X_test_fold, y_test_fold,
            isacore_Embeddings = {}, affectiveSpace_Embeddings = {}):


    if boolDict["KI"]:
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


    if emb_type == "ConceptNet":
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


    if boolDict["KI"]:
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

    # Applying RandomOverSampler to training set
    # Random seed set for now to make sure all pipelines overfit the same way
    if boolDict["SMOTE"]:
        valueCounts = pd.Series(y_train_fold).value_counts()
        print("Before:", valueCounts)
        # y_train_fold.value_counts()
        over = RandomOverSampler(random_state=SMOTE_random_seed)
        steps = [('o', over)]
        pipeline = Pipeline(steps=steps)
        if emb_type == "BERT":
            train_input_ids, _ = pipeline.fit_resample(train_input_ids, y_train_fold)
            if not boolDict["KI"]:
                train_attention_masks, y_train_fold = pipeline.fit_resample(train_attention_masks, y_train_fold)
            else:
                train_attention_masks, _ = pipeline.fit_resample(train_attention_masks, y_train_fold)
                isa_train_tokens, _ = pipeline.fit_resample(isa_train_tokens, y_train_fold)
                aff_train_tokens, y_train_fold = pipeline.fit_resample(aff_train_tokens, y_train_fold)

        if emb_type == "ConceptNet":
            if not boolDict["KI"]:
                con_train_tokens, y_train_fold = pipeline.fit_resample(con_train_tokens, y_train_fold)
            else:
                con_train_tokens, _ = pipeline.fit_resample(con_train_tokens, y_train_fold)
                isa_train_tokens, _ = pipeline.fit_resample(isa_train_tokens, y_train_fold)
                aff_train_tokens, y_train_fold = pipeline.fit_resample(aff_train_tokens, y_train_fold)
            print("After:", pd.Series(y_train_fold).value_counts())

    # Convert to BERT embeddings for BERT models
    if emb_type == "BERT":
        # text, tokenizer, model, 2d or 3d embeddings
        X_train_emb_fold = getFeatures(text=X_train_fold, tokenizer=tokenizer, model=model, model_name=model_name,
                                       max_length=max_length, return_tensor_type="pt")
        X_test_emb_fold = getFeatures(text=X_test_fold, tokenizer=tokenizer, model=model, model_name=model_name,
                                      max_length=max_length, return_tensor_type="pt")
        number_channels, number_features = X_train_emb_fold.shape[1], X_train_emb_fold.shape[2]
    elif emb_type == "ConceptNet":
        # Rename tokens for ConceptNet models for simplicity of coding
        X_train_emb_fold = con_train_tokens
        X_test_emb_fold = con_test_tokens
        number_channels, number_features = 512, 768

    X_train_emb_fold = tf.convert_to_tensor(X_train_emb_fold)
    X_test_emb_fold = tf.convert_to_tensor(X_test_emb_fold)

    if boolDict["KI"]:
        isa_train_tokens = tf.convert_to_tensor(isa_train_tokens)
        isa_test_tokens = tf.convert_to_tensor(isa_test_tokens)
        aff_train_tokens = tf.convert_to_tensor(aff_train_tokens)
        aff_test_tokens = tf.convert_to_tensor(aff_test_tokens)

    if boolDict["KI"]:
        modelTrain = [X_train_emb_fold, isa_train_tokens, aff_train_tokens]
        modelTest = [X_test_emb_fold, isa_test_tokens, aff_test_tokens]
    else:
        modelTrain = X_train_emb_fold
        modelTest = X_test_emb_fold


    if boolDict["KI"] == True and emb_type == "ConceptNet":
        vocab_sizes = [con_vocab_size, isa_vocab_size, aff_vocab_size]
        embed_matrices = [concept_embedding_matrix, isa_embedding_matrix, affectiveSpace_embedding_matrix]
    elif boolDict["KI"] == False and emb_type == "ConceptNet":
        vocab_sizes = [con_vocab_size]
        embed_matrices = [concept_embedding_matrix]
    elif boolDict["KI"] == True and emb_type == "BERT":
        vocab_sizes = [isa_vocab_size, aff_vocab_size]
        embed_matrices = [isa_embedding_matrix, affectiveSpace_embedding_matrix]
    elif boolDict["KI"] == False and emb_type == "BERT":
        vocab_sizes = []
        embed_matrices = []

    if boolDict["weight"]:
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


    if boolDict["KI"]:
        checkpointName = f"{emb_type}_{modelType}_with_KI_best_model.h5"
    else:
        checkpointName = f"{emb_type}_{modelType}_no_KI_best_model.h5"

    es = EarlyStopping(monitor='val_auc', mode="max", patience=10, min_delta=0)
    mc = ModelCheckpoint(checkpointName, monitor='val_auc', mode='max', verbose=0,
                         save_best_only=True)

    if boolDict["tuning"]:
        param_grid = hyperparameters
        tpe_trials = Trials()
        tpe_best = []

        if modelType == "CNN":
            objectiveFunc = partial(objectiveFunctionCNN, num_channels=number_channels,num_features=number_features,
                                    Xtrain=modelTrain, ytrain=y_train_fold, Xtest=modelTest, ytest=y_test_fold,
                                    num_label=num_labels, modelType = modelType, e_type=emb_type, max_length=max_length,
                                    know_infus_bool=boolDict["KI"], vocabSize=vocab_sizes, embedding_matrix=embed_matrices,
                                    es=es, mc=mc)
        elif modelType == "GRU" or modelType == "LSTM":
            objectiveFunc = partial(objectiveFunctionRNN, num_channels=number_channels, num_features=number_features,emb_dim=embed_dimen,
                                    Xtrain=modelTrain, ytrain=y_train_fold, Xtest=modelTest, ytest=y_test_fold,
                                    n_lab=num_labels, e_type=emb_type, modelType = modelType, max_length=max_length,
                                    know_infus_bool=boolDict["KI"], vocabSize=vocab_sizes, embedding_matrix=embed_matrices,
                                    es=es, mc=mc, model_name=model_name)

        tpe_best = fmin(fn=objectiveFunc, space=param_grid, algo=tpe.suggest,
                        max_evals=global_max_evals, trials=tpe_trials)
        hyperparameters = space_eval(param_grid, tpe_best)
        print("Best: ", getXfromBestModelfromTrials(tpe_trials, 'loss'), hyperparameters)

    print("--------------------")
    print(hyperparameters)
    print("--------------------")

    if modelType == "CNN":
        nnModel = cnnModel(hyperparameters, number_channels, number_features, num_labels, emb_type, boolDict["KI"],
                           max_length, vocab_sizes, embed_matrices)
    elif modelType == "GRU" or modelType == "LSTM":
        nnModel = RNNModel(hyperparameters, number_channels, number_features, num_labels, emb_type, model_name,
                           modelType, boolDict["KI"], embed_dimen, preTrainDim, max_length, vocab_sizes, embed_matrices)

    if boolDict["weight"]:
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

def run(outputPath, UMD_path, CSSRS_path, model_name, mlm_params, transferLearning, mlm_pretrain, CSSRS_n_label, boolDict,
        embed_dimen, hyperparameters, n_folds, modelType, max_length):
    if mlm_pretrain:
        tokenizer, model = getModel(model_name)

        task_train, task_test = importUMD(UMD_path, "crowd", "A")
        # For now, remove all posts with empty post_body (i.e. just post title)
        task_train = task_train[task_train["post_body"].notnull()]  # Removes 72 posts
        task_test = task_test[task_test["post_body"].notnull()]  # Removes 5 posts
        # Sort user_id and timestamp in descending order for both
        task_train = task_train.sort_values(by=["user_id", "timestamp"], ascending=[True, True], ignore_index=True)
        task_test = task_test.sort_values(by=["user_id", "timestamp"], ascending=[True, True], ignore_index=True)
        task_train["model_input"] = task_train["post_body"].map(lambda x: MaskingFunction(x, tokenizer=tokenizer,
                                                                                          custom_masking=False,
                                                                                          max_length=512,
                                                                                          return_tensor_type="pt"))
        task_test["model_input"] = task_test["post_body"].map(lambda x: MaskingFunction(x, tokenizer=tokenizer,
                                                                                        custom_masking=False,
                                                                                        max_length=512,
                                                                                        return_tensor_type="pt"))

        train_model_input = {key: torch.stack([i[key][0] for i in task_train["model_input"]]) for key in
                             task_train["model_input"][0]}
        test_model_input = {key: torch.stack([i[key][0] for i in task_test["model_input"]]) for key in
                            task_test["model_input"][0]}

        trainDataset = customDataset(train_model_input)
        testDataset = customDataset(test_model_input)

        ## From https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # and move our model over to the selected device
        model.to(device)
        # initialize optimizer
        optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

        # activate training mode
        model.train()
        loader = torch.utils.data.DataLoader(trainDataset, batch_size=mlm_params["batch_size"], shuffle=True)
        for epoch in range(mlm_params["epochs"]):
            # setup loop with TQDM and dataloader
            loop = tqdm(loader, leave=True)
            for batch in loop:
                # initialize calculated gradients (from prev step)
                optim.zero_grad()

                loss = runModel(batch, model, device)
                # calculate loss for every parameter that needs grad update
                loss.backward()
                # update parameters
                optim.step()
                # print relevant info to progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

        model.save_pretrained(fr"D:\zProjects\MLM\Saved_Models\UMD_MLM_pretrain_{model_name}")

    ##-----Regular modeling-----
    if transferLearning:
        model_path = fr"D:\zProjects\MLM\Saved_Models\UMD_MLM_pretrain_{model_name}"
        if model_name == "BERT":
            if os.path.exists(model_path):
                model = BertModel.from_pretrained(fr"D:\zProjects\MLM\Saved_Models\UMD_MLM_pretrain_{model_name}")
            else:
                print("No transfer learning applied. Loading from huggingface checkpoint.")
                model = getRegularModel(model_name)
        elif model_name == "ROBERTA":
            if os.path.exists(model_path):
                model = RobertaModel.from_pretrained(fr"D:\zProjects\MLM\Saved_Models\UMD_MLM_pretrain_{model_name}")
            else:
                print("No transfer learning applied. Loading from huggingface checkpoint.")
                model = getRegularModel(model_name)
        elif model_name == "ELECTRA":
            if os.path.exists(model_path):
                model = ElectraModel.from_pretrained(fr"D:\zProjects\MLM\Saved_Models\UMD_MLM_pretrain_{model_name}")
            else:
                print("No transfer learning applied. Loading from huggingface checkpoint.")
                model = getRegularModel(model_name)
        model.eval()
        tokenizer = getTokenizer(model_name)

        if tf.test.gpu_device_name():
            print('GPU: {}'.format(tf.test.gpu_device_name()))
        else:
            print('CPU version')
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        startTime = datetime.now()
        CSSRS = importCSSRS(CSSRS_path, num_labels=CSSRS_n_label)
        # Make all text lowercase
        text = CSSRS["Post"].apply(lambda x: x.lower())
        labels = CSSRS["Label"]

        # If spittling entire dataset into train-test split
        if boolDict["split"] == True:
            test_size = 0.25
            X_train, X_test, y_train, y_test = train_test_split(text, labels, test_size=test_size, shuffle=True,
                                                                stratify=labels, random_state=split_random_seed)
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            df = pd.DataFrame({"Post": list(X_train), "Label": y_train}, columns=['Post', 'Label'])
        else:
            df = pd.DataFrame({"Post": text, "Label": labels}, columns=['Post', 'Label'])

        # Create alias for principal embedding
        # Holdover from previous code. Will remove once I remove all the principal conceptnet embedding code
        emb_type = "BERT"

        if boolDict["CV"] == True:
            # Define per-fold score containers
            acc_per_fold = []
            loss_per_fold = []
            fold_stats = []
            fold_matrix = []
            fold_hyper = []
            test_size_per_fold = []

            whole_results = pd.DataFrame({"Actual": pd.Series(dtype=int), "Predicted": pd.Series(dtype=int),
                                          "PredictedProba": pd.Series(dtype=int), "Fold": pd.Series(dtype=int)})
            fold_results = []
            sfk = StratifiedKFold(n_splits=n_folds, shuffle=False)
            fold_num = 1
            for train_indx, test_indx in sfk.split(df["Post"], df["Label"]):

                fold_train = df.iloc[train_indx].copy()
                fold_train = fold_train.reset_index(drop=True)
                X_train_fold = fold_train["Post"]
                y_train_fold = fold_train["Label"]

                fold_test = df.iloc[test_indx].copy()
                fold_test = fold_test.reset_index(drop=True)
                X_test_fold = fold_test["Post"]
                y_test_fold = fold_test["Label"]

                y_train_fold = tf.convert_to_tensor(y_train_fold)
                y_test_fold = tf.convert_to_tensor(y_test_fold)
                # Generate a print
                print('------------------------------------------------------------------------')
                print(f'Training for fold {fold_num} ...')

                if boolDict["KI"]:
                    nnModel, history, scores, \
                    y_pred_proba, hyperparameters = runFold(outputPath=outputPath, filespath=CSSRS_path,
                                                            model=model, tokenizer=tokenizer,
                                                            model_name=model_name, modelType=modelType,
                                                            emb_type=emb_type, max_length=max_length,
                                                            num_labels=CSSRS_n_label, embed_dimen=embed_dimen,
                                                            hyperparameters=hyperparameters, n_folds=n_folds,
                                                            fold_num=fold_num, X_train_fold=X_train_fold,
                                                            y_train_fold=y_train_fold, X_test_fold=X_test_fold,
                                                            y_test_fold=y_test_fold, boolDict=boolDict,
                                                            isacore_Embeddings=isacore_Embeddings,
                                                            affectiveSpace_Embeddings=affectiveSpace_Embeddings)
                else:
                    nnModel, history, scores, \
                    y_pred_proba, hyperparameters = runFold(outputPath=outputPath, filespath=CSSRS_path,
                                                            model=model, tokenizer=tokenizer,
                                                            model_name=model_name, modelType=modelType,
                                                            emb_type=emb_type, max_length=max_length,
                                                            num_labels=CSSRS_n_label, embed_dimen=embed_dimen,
                                                            hyperparameters=hyperparameters, n_folds=n_folds,
                                                            fold_num=fold_num, X_train_fold=X_train_fold,
                                                            y_train_fold=y_train_fold, X_test_fold=X_test_fold,
                                                            y_test_fold=y_test_fold, boolDict=boolDict)

                train_auc = history.history['auc']
                val_auc = history.history['val_auc']
                train_acc = history.history['accuracy']
                val_acc = history.history['val_accuracy']
                train_loss = history.history['loss']
                val_loss = history.history['val_loss']
                epochs = range(len(train_auc))

                fold_results.append(
                    {"train_auc": train_auc, "val_auc": val_auc, "train_acc": train_acc, "val_acc": val_acc,
                     "train_loss": train_loss, "val_loss": val_loss, "epochs": epochs})

                # Generate generalization metrics
                print(
                    f'Score for fold {fold_num}: {nnModel.metrics_names[0]} of {scores[0]}; {nnModel.metrics_names[1]} of {scores[1] * 100}%')
                acc_per_fold.append(scores[1] * 100)
                loss_per_fold.append(scores[0])
                fold_hyper.append(hyperparameters)

                y_pred = np.argmax(y_pred_proba, axis=1)
                whole_results = pd.concat(
                    [whole_results, pd.DataFrame({"Actual": y_test_fold.numpy().tolist(), "Predicted": y_pred.tolist(),
                                                  "PredictedProba": y_pred_proba.tolist(), "Fold": fold_num})],ignore_index=True)

                print(classification_report(y_test_fold, y_pred))

                # contains precision, recall, and f1 score for each class
                report = classification_report(y_test_fold, y_pred, output_dict=True)

                # Get only precision, recall, f1-score, and support statistics
                # filtered_report = {str(label): report[str(label)] for label in range(num_labels)}

                matrix = confusion_matrix(y_test_fold, y_pred)
                print(f"{CSSRS_n_label}-label confusion matrix")
                print(matrix)
                # Increase Fold Number
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

            overallResults = getStatistics(outputPath, whole_results["Actual"], whole_results["PredictedProba"],
                                           whole_results["Predicted"], CSSRS_n_label)
            # whole_results.to_csv(os.path.join(outputPath, "Actual_vs_Predicted.csv"), index=False)
            endTime = datetime.now()
            elapsedTime = endTime - startTime

            printOverallResults(outputPath=outputPath, fileName=f"OverallResults {CSSRS_n_label}Label.csv",
                                n_label=CSSRS_n_label, emb_type=emb_type, max_length=max_length, boolDict=boolDict,
                                numCV=n_folds, model_type=modelType, stats=overallResults,
                                hyperparameters=fold_hyper, execTime=elapsedTime, whole_results=whole_results,
                                fold_results=fold_results, model_name=model_name)
        else:
            y_train = tf.convert_to_tensor(y_train)
            y_test = tf.convert_to_tensor(y_test)

            if boolDict["KI"] == True:
                nnModel, history, scores, y_pred_proba, hyperparameters = runFold(outputPath=outputPath, filespath=CSSRS_path,
                                                                                  model=model, tokenizer=tokenizer,
                                                                                  model_name=model_name, modelType=modelType,
                                                                                  emb_type=emb_type,
                                                                                  max_length=max_length, num_labels=CSSRS_n_label,
                                                                                  embed_dimen=embed_dimen,
                                                                                  hyperparameters=hyperparameters,
                                                                                  n_folds=n_folds, fold_num=1,
                                                                                  X_train_fold=X_train, y_train_fold=y_train,
                                                                                  X_test_fold=X_test, y_test_fold=y_test,
                                                                                  boolDict=boolDict,
                                                                                  isacore_Embeddings=isacore_Embeddings,
                                                                                  affectiveSpace_Embeddings=affectiveSpace_Embeddings)
            else:
                nnModel, history, scores, y_pred_proba, hyperparameters = runFold(outputPath=outputPath, filespath=CSSRS_path,
                                                                                  model=model, tokenizer=tokenizer,
                                                                                  model_name=model_name, modelType=modelType,
                                                                                  emb_type=emb_type,
                                                                                  max_length=max_length, num_labels=CSSRS_n_label,
                                                                                  embed_dimen=embed_dimen,
                                                                                  hyperparameters=hyperparameters,
                                                                                  n_folds=n_folds, fold_num=1,
                                                                                  X_train_fold=X_train, y_train_fold=y_train,
                                                                                  X_test_fold=X_test, y_test_fold=y_test,
                                                                                  boolDict=boolDict)
            y_pred = np.argmax(y_pred_proba, axis=1)

            whole_results = pd.DataFrame({"Actual": y_test.numpy().tolist(), "Predicted": y_pred.tolist(),
                                          "PredictedProba": y_pred_proba.tolist()})

            overallResults = getStatistics(outputPath, whole_results["Actual"], whole_results["PredictedProba"],
                                           whole_results["Predicted"], CSSRS_n_label)
            fold_results = []
            # printPredictions(y_test, y_pred, num_labels, outputPath)
            endTime = datetime.now()
            elapsedTime = endTime - startTime

            printOverallResults(outputPath=outputPath, fileName=f"OverallResults {CSSRS_n_label}Label (no CV).csv",
                                n_label=CSSRS_n_label, emb_type=emb_type, max_length=max_length, boolDict=boolDict,
                                numCV=n_folds, model_type=modelType, stats=overallResults,
                                hyperparameters=hyperparameters,execTime=elapsedTime, whole_results=whole_results,
                                fold_results=fold_results, model_name=model_name)

def main():
    if platform.system() == "Windows":
        filePath = r"D:\Summer 2022 Project\Reddit C-SSRS\500_Reddit_users_posts_labels.csv"
        outputPath = r"D:\zProjects\MLM\Output\CSSRS"

        UMD_path = r"C:\Users\dmlee\Desktop\Summer_Project\Fall 2022\umd_reddit_suicidewatch_dataset_v2"
        CSSRS_path = r"D:\Summer 2022 Project\Reddit C-SSRS\500_Reddit_users_posts_labels.csv"
    elif platform.system() == "Linux":
        filePath = r"/ddn/home12/r3102/files/500_Reddit_users_posts_labels.csv"
        outputPath = r"/ddn/home12/r3102/results"

    # All models currently implemented are the base uncased versions
    # model_name = "BERT"
    model_name = "ROBERTA"
    # model_name = "ELECTRA"

    # modType = "CNN"
    modType = "GRU"
    # modType = "LSTM"

    # mlm_pretrain = True
    mlm_pretrain = False

    transferLearn = True
    # transferLearn = False


    CSSRS_n_labels = 4
    number_of_folds = 5
    max_length = 512

    mlm_params = {"epochs":10, "batch_size":16}

    splitBool = False
    CVBool = True
    know_infuse = False
    SMOTE_bool = False
    weight_bool = True

    parameter_tune = False
    if parameter_tune == True:
        param_grid = {"epochs": hp.choice("epochs", [10, 25, 50]),
                      "batch_size": hp.choice("batch_size", [4, 24, 32]),
                      "dropout": hp.choice("droupout", [0.1, 0.2, 0.3, 0.4, 0.5]),
                      "learning_rate":hp.choice("learning_rate", [0.01, 0.005, 0.001]),
                      "rnn_nodes": hp.choice("rnn_nodes", [128, 256])}
    else:                                        #Default Values
        param_grid = {"batch_size": 32,          #32, 4 for original CNN
                      "dropout": 0.25,           #0.25 for RNN, 0.3 for CNN
                      "epochs": 10,              #10, 50 for original CNN
                      "learning_rate":0.001,     #0.001
                      "rnn_nodes":128,           #128
                      "1st_dense":300,           #300
                      "2nd_dense":100}           #100

    boolDict = {"split":splitBool, "CV":CVBool, "KI":know_infuse,
                "SMOTE":SMOTE_bool, "weight":weight_bool, "tuning":parameter_tune}

    #embed_dimen holdover from previous code. Will eventually remove after cleaning up old references
    run(outputPath=outputPath,UMD_path=UMD_path, CSSRS_path=CSSRS_path, model_name=model_name, mlm_params=mlm_params,
        mlm_pretrain=mlm_pretrain, transferLearning=transferLearn, CSSRS_n_label=CSSRS_n_labels, boolDict=boolDict, hyperparameters=param_grid,
        n_folds= number_of_folds, modelType=modType, max_length=max_length,
        embed_dimen="3D")


global_max_evals = 30
if platform.system() == "Windows":
    preprocessor_link = r"https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
    encoder_link = r"https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

    numberbatch_path = r"D:\Summer 2022 Project\numberbatch-en.txt"
    isacore_path = r"D:\Summer 2022 Project\isacore\isacore.csv"
    affectiveSpace_path = r"D:\Summer 2022 Project\affectivespace\affectivespace.csv"
elif platform.system() == "Linux":
    preprocessor_link = r"/ddn/home12/r3102/files/TF_BERT/bert_en_uncased_preprocess_3"
    encoder_link = r"/ddn/home12/r3102/files/TF_BERT/bert_en_uncased_L-12_H-768_A-12_4"

    numberbatch_path = r"/ddn/home12/r3102/files/numberbatch-en.txt"
    isacore_path = r"/ddn/home12/r3102/files/isacore.csv"
    affectiveSpace_path = r"/ddn/home12/r3102/files/affectivespace.csv"

# glove_100d_path = r"D:\Summer 2022 Project\glove.6B.100d.txt"
preTrainDim = 300
seed = 99
newOutputFolder = r"C:\Users\dmlee\Desktop\Summer_Project\Summer 2022\Output\CSSRS\temp"
main()

