#!/usr/bin/env python3

from Libraries import *

def getXfromBestModelfromTrials(trials, x):
    valid_trial_list = [trial for trial in trials if STATUS_OK == trial['result']['status']]
    losses = [float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result'][x]

def bert_encode(data, tokenizer, maximum_length):
    encoded = data.apply(lambda x: tokenizer.encode_plus(x, add_special_tokens=True, truncation=True,
                                                           max_length=maximum_length,padding="max_length",
                                                           return_attention_mask=True))

    input_ids = encoded.apply(lambda x: x['input_ids'])
    input_ids = np.array(input_ids.values.tolist())
    attention_masks = encoded.apply(lambda x: x['attention_mask'])
    attention_masks = np.array(attention_masks.values.tolist())
    return input_ids, attention_masks

def getEmbeddings(input_ids, attention_mask,  model, embd_dimen):
    """
       parameters
       ------------------
       padded
           numpy array of padded to sequences
        model
            transformers model
        embed_dimen
            string determining whether to return 2d or 3d embeddings
       """

    # input_word_ids = tf.keras.layers.Input(shape=(512,), name='input_token', dtype='int32')
    # input_mask = tf.keras.layers.Input(shape=(512,), name='masked_token', dtype='int32')
    # X = model(input_ids, input_masks_ids)[0]
    # pooled_output, sequence_output = model([input_word_ids, input_mask])
    # nMod = Model(inputs=[input_word_ids, input_mask], outputs=[pooled_output, sequence_output])

    # model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs=X)(X)
    # y = nMod.predict([input_ids, attention_mask])
    # print(y.shape)
    # print(y)
    # input("WAIT")
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    if embd_dimen == "2d":
        features = last_hidden_states[0][:, 0, :].numpy()  # use this line if you want the 2D BERT features
    elif embd_dimen == "3d":
        features = last_hidden_states[0].numpy() # use this line if you want the 3D BERT features

    return features

def getTokens(standardization, splitMethod, maxTokens, outputLength, vocab, num_ngrams, Xtrain, Xtest):
    """
    parameters
    ------------------
    standardization
        standardization method for TextVectorization
    splitMethod
        splitting method for TextVectorization
    maxTokens
        maximum number of tokens for TextVectorization
    outputLength
        output of sequence of tokens for TextVectorization
    vocab
        vocabulary to build on for TextVectorization
    num_ngrams
        number of token ngrams to generate for TextVectorization
    Xtrain
        training features to tokenize
    Xtest
        testing features to tokenize
    """
    vectorize_layer = TextVectorization(standardize=standardization, split=splitMethod,
                                            max_tokens=maxTokens, output_sequence_length=outputLength, vocabulary=vocab,
                                            ngrams=num_ngrams)

    voc = vectorize_layer.get_vocabulary()
    vocab_size = vectorize_layer.vocabulary_size()
    word_index = dict(zip(voc, range(vocab_size)))

    t = Sequential()
    t.add(Input(shape=(1,), dtype=tf.string))
    t.add(vectorize_layer)
    train_tokens = t.predict(Xtrain)
    test_tokens = t.predict(Xtest)

    return vocab_size, word_index, train_tokens, test_tokens

def extractList(df):
    """
       parameters
       ------------------
       df
           dataframe where column contains strings of a list
       """
    df.loc[df["Post"].str.endswith("]") == False, "Post"] = df.loc[df["Post"].str.endswith("]") == False, "Post"] + "']"
    df["Post"] = df["Post"].apply(lambda x: ast.literal_eval(x))

    # extract_list = df["Post"].apply(lambda x: [x.strip("'").strip("'") for x in x.strip('][').split('\', ')])

def onehotEncode(labels):
    onehot_encoded = list()
    for value in labels:
        encoded = [0 for _ in range(max(labels) + 1)]
        encoded[value] = 1
        onehot_encoded.append(encoded)
    new_labels = convert_to_tensor(onehot_encoded)
    return new_labels

def multiclass_accuracy(matrix, num_classes):
    return {str(i): {"acc": (matrix.diagonal() / matrix.sum(axis=1)).tolist()[i]} for i in range(num_classes)}

def multiclass_ROC_AUC(y_test, y_pred_proba, n_classes):
    fpr = []
    tpr = []
    roc_auc = {}
    for i in range(n_classes):
        df_aux = pd.DataFrame()
        df_aux['class'] = [1 if y == i else 0 for y in y_test]
        df_aux['prob'] = y_pred_proba[:, i]
        df_aux = df_aux.reset_index(drop=True)
        fpr_i, tpr_i, _ = roc_curve(df_aux['class'], df_aux['prob'])
        fpr.append(fpr_i)
        tpr.append(tpr_i)
        aoc_val = auc(fpr[i], tpr[i])
        roc_auc[str(i)] = {"auc": aoc_val}
    return(roc_auc)

def mergeDicts(list_of_dicts):
    first_dict = list_of_dicts[0]
    for i in range(1, len(list_of_dicts)):
        for label, statistics in list_of_dicts[i].items():
            for statistic, value in statistics.items():
                first_dict[label][statistic] = value
    return(first_dict)

def printOverallResults(outputPath, fileName, n_label, emb_type, max_length, SMOTE_bool, splitBool, numCV, model_type,
                        know_infus_bool, parameter_tune_bool, stats, hyperparameters, execTime, whole_results, fold_results):
    if numCV != 5:
        outputPath = os.path.join(outputPath, f"[{numCV} Folds]")

    hours, minutes, seconds = str(execTime).split(":")
    results = pd.DataFrame({"Number of labels":n_label, "Embedding":emb_type, "Max Sentence Length":max_length,
                            "Model":model_type, "Knowledge Infusion": know_infus_bool, "SMOTE":SMOTE_bool,
                            "Parameter Tuning":parameter_tune_bool ,"CV Folds":numCV,
                            "Macro Average":stats["accuracy"],"Precision":stats["macro avg"]["precision"],
                            "Recall":stats["macro avg"]["recall"],"F1-score":stats["macro avg"]["f1-score"],
                            "AUC":stats["auc"], "New False Positive Rate":stats["graded_fp"],
                            "New False Negative Rate":stats["graded_fn"], "Ordinal Error":stats["ordinal_err"],
                            "Execution Time":f"{hours}H{minutes}M", "random.seed":seed, "np seed":seed, "tf seed":seed,
                            "train_test split seed":split_random_seed, "SMOTE seed":SMOTE_random_seed,
                            "KFold seed":KFold_shuffle_random_seed}, index=[0])
    for i in range(n_label):
        results[f"Label {i} Accuracy"]  = stats[str(i)]["acc"]

    if splitBool == True:
        results[f"Hyperparameters"] = str(sorted(list(hyperparameters[i].items()), key=lambda x: x[0][0]))
    else:
        for i in range(numCV):
            results[f"Fold {i+1} Hyperparameters"] = str(sorted(list(hyperparameters[i].items()), key=lambda x: x[0][0]))

    file_path = os.path.join(outputPath, fileName)

    if not os.path.exists(file_path):
        qid = 1
    else:
        temp_df = pd.read_csv(file_path)
        qid = temp_df.iloc[-1,0] + 1
    results["QID"] = qid


    if numCV == 5:
        if n_label == 4:
            results = results[["QID", "Number of labels", "Embedding", "Max Sentence Length", "Model", "Knowledge Infusion",
                               "SMOTE", "Parameter Tuning", "CV Folds","Label 0 Accuracy", "Label 1 Accuracy","Label 2 Accuracy","Label 3 Accuracy",
                               "Macro Average","Precision", "Recall","F1-score", "AUC", "New False Positive Rate","New False Negative Rate",
                           "Ordinal Error", "Fold 1 Hyperparameters", "Fold 2 Hyperparameters", "Fold 3 Hyperparameters",
                               "Fold 4 Hyperparameters","Fold 5 Hyperparameters", "Execution Time", "random.seed", "np seed",
                               "tf seed", "train_test split seed", "SMOTE seed", "KFold seed"]]
        elif n_label == 5:
            results = results[["QID", "Number of labels", "Embedding", "Max Sentence Length", "Model", "Knowledge Infusion",
                               "SMOTE", "Parameter Tuning", "CV Folds","Label 0 Accuracy", "Label 1 Accuracy", "Label 2 Accuracy", "Label 3 Accuracy",
                               "Label 4 Accuracy","Macro Average", "Precision", "Recall", "F1-score", "AUC", "New False Positive Rate",
                           "New False Negative Rate", "Ordinal Error", "Fold 1 Hyperparameters", "Fold 2 Hyperparameters",
                               "Fold 3 Hyperparameters","Fold 4 Hyperparameters","Fold 5 Hyperparameters", "Execution Time",
                               "random.seed", "np seed", "tf seed", "train_test split seed", "SMOTE seed", "KFold seed"]]

    file_path = os.path.join(outputPath, fileName)
    results.to_csv(file_path, mode="a", index=False, header = not os.path.exists(file_path))


    if splitBool == True:
        actual_vs_pred = f"(No CV) {emb_type}, Max length of {max_length},{' No' if SMOTE_bool == False else ''} SMOTE, {model_type}, " \
                           f"{'No' if know_infus_bool == False else 'with'} Knowledge Infusion, Actual_vs_Predicted"
        conf_matrix_name = f"(No CV) {emb_type}, Max length of {max_length},{' No' if SMOTE_bool == False else ''} SMOTE, {model_type}, " \
                           f"{'No' if know_infus_bool == False else 'with'} Knowledge Infusion, Confusion Matrix"
    else:
        actual_vs_pred = f"[{qid}] {emb_type}, Max length of {max_length},{' No' if SMOTE_bool == False else ''} SMOTE, {model_type}, " \
                           f"{'No' if know_infus_bool == False else 'with'} Knowledge Infusion, {numCV} Folds, Actual_vs_Predicted"
        conf_matrix_name = f"[{qid}] {emb_type}, Max length of {max_length},{' No' if SMOTE_bool == False else ''} SMOTE, {model_type}, " \
                           f"{'No' if know_infus_bool == False else 'with'} Knowledge Infusion, {numCV} Folds, Confusion Matrix"
    folder = os.listdir(outputPath)

    #Check if file with same name exists
    #If so, add the appropriate number at the end to designate different results using same parameters
    file_count = 0
    for file in folder:
        if conf_matrix_name in file:
            file_count = file_count + 1

    matDF = pd.DataFrame(stats["matrix"], index=[i for i in range(n_label)], columns=[i for i in range(n_label)])
    ax = sns.heatmap(matDF, annot=True, cmap="Blues", fmt='d').get_figure()
    if file_count == 0:
        ax.savefig(os.path.join(outputPath, conf_matrix_name + f".png"))
        whole_results.to_csv(os.path.join(outputPath, actual_vs_pred + ".csv"), index=False)
    else:
        ax.savefig(os.path.join(outputPath,conf_matrix_name + f" ({file_count}).png"))
        whole_results.to_csv(os.path.join(outputPath, actual_vs_pred + f" ({file_count}).csv"), index=False)


    # Fold training/validation statistics
    for i in range(len(fold_results)):
        fig = plt.figure(figsize=(15, 15))
        ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=1, colspan=1)
        ax1 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, colspan=1)

        ax0.plot(fold_results[i]['epochs'], fold_results[i]['train_acc'], color='red', marker='o',
                 linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['acc_v'], legend='brief', label="accuracy")
        ax0.plot(fold_results[i]['epochs'], fold_results[i]['val_acc'], color='green', marker='o',
                 linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['val_acc'], legend='brief', label="val_accuracy")
        ax0.plot(fold_results[i]['epochs'], fold_results[i]['train_loss'], color='black', marker='o',
                 linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['loss_v'], legend='brief', label="loss")
        ax0.plot(fold_results[i]['epochs'], fold_results[i]['val_loss'], color='blue', marker='o',
                 linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['val_loss'], legend='brief', label="val_loss")
        ax0.set_title(f'Fold {i+1}: Training and validation accuracy/loss')  # , y=1.05, size=15)
        ax0.legend(['train_acc', 'val_acc', 'train_loss', 'val_loss'])

        ax1.set_title(f'Fold {i+1}: Training and validation AUC')
        ax1.plot(fold_results[i]['epochs'], fold_results[i]['train_auc'], color='red', marker='o',
                 linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['acc_v'], legend='brief', label="accuracy")
        ax1.plot(fold_results[i]['epochs'], fold_results[i]['val_auc'], color='green', marker='o',
                 linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['val_acc'], legend='brief', label="val_accuracy")
        ax1.legend(['train_auc', 'val_auc'])
        fig.savefig(os.path.join(outputPath, f" [{qid}] Fold  {i+1} - Training and Validation Accuracy, Loss, AUC.png"))

        # plt.tight_layout()
        # plt.show()

def getLabels(df, num_labels):
    scale_conversion4 = {"Supportive": "No-risk",
                         "Indicator": "No-risk",
                         "Ideation": "Ideation",
                         "Behavior": "Behavior",
                         "Attempt": "Attempt"}

    label_conversion = {"Supportive": 0,
                        "Indicator": 1,
                        "Ideation": 2,
                        "Behavior": 3,
                        "Attempt": 4}

    label_conversion4 = {"No-risk": 0,
                         "Ideation": 1,
                         "Behavior": 2,
                         "Attempt": 3}

    if num_labels == 5:
        df = df.replace({"Label": label_conversion})
        inv_map = {val: key for key, val in label_conversion.items()}
        return (df, inv_map)
    elif num_labels == 4:
        df = df.replace({"Label": scale_conversion4})
        df = df.replace({"Label": label_conversion4})
        inv_map = {val: key for key, val in label_conversion4.items()}
        return(df, inv_map)


def getStatistics(outputPath, y_test, y_pred_proba, y_pred, n_lab):
    y_pred_proba = np.array(y_pred_proba.tolist())

    # Multi-label
    print(classification_report(y_test, y_pred))
    # contains precision, recall, and f1 score for each class
    report = classification_report(y_test, y_pred, output_dict=True)

    # Get only precision, recall, f1-score, and support statistics
    # filtered_report = {str(label): report[str(label)] for label in range(y_test.nunique())}

    matrix = confusion_matrix(y_test, y_pred)
    print(f"{n_lab}-label confusion matrix")
    print(matrix)

    label_accuracies = multiclass_accuracy(matrix, n_lab)
    class_auc = multiclass_ROC_AUC(y_test, y_pred_proba, n_lab)
    aucValue = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average='macro')

    new_fp = (sum(y_pred > y_test.to_numpy())) / len(y_pred)
    new_fn = (sum(y_pred < y_test.to_numpy())) / len(y_pred)
    oe = (sum(abs(y_pred - y_test.to_numpy()) > 1)) / len(y_pred)
    print(f"New False Positive Rate = {new_fp}")
    print(f"New False Negative Rate = {new_fn}")
    print(f"New Ordinal Error = {oe}")

    # argument is list of dictionaries to merge
    class_statistics = mergeDicts([report, label_accuracies, class_auc])
    class_statistics["matrix"] = matrix
    class_statistics["auc"] = aucValue
    class_statistics["graded_fp"] = new_fp
    class_statistics["graded_fn"] = new_fn
    class_statistics["ordinal_err"] = oe

    return class_statistics

def printPredictions(y_test, y_pred, n_lab, outputPath):
    if n_lab == 4:
        df = pd.DataFrame({"Actual":y_test, "Pred Label":y_pred})
        df.to_csv(os.path.join(outputPath, "PredictedLabels (4-Label).csv"))
    elif n_lab == 5:
        df = pd.DataFrame({"Actual": y_test, "Pred Label": y_pred})
        df.to_csv(os.path.join(outputPath, "PredictedLabels (4-Label).csv"))

def getSummStats(CSSRS):
    # return
    #Raw number of each label in 5-label schema
    print("Frequency of each label in 5-label schema")
    print(CSSRS.value_counts("Label"))

    #Percentage of each label in 5-label schema
    print("Percentage of each label in 5-label schema")
    CSSRS["Label"].value_counts(normalize=True) * 100


    #Convert from 5-label to 4-label
    scale_conversion4 = {"Supportive": "No-risk",
                         "Indicator": "No-risk",
                         "Ideation": "Ideation",
                         "Behavior": "Behavior",
                         "Attempt": "Attempt"}

    CSSRS_4 = CSSRS.replace({"Label":scale_conversion4})

    #Raw number of each label in 4-label schema
    print("Frequency of each label in 4-label schema")
    print(CSSRS_4.value_counts("Label"))

    #Percentage of each label in 4-label schema
    print("Percentage of each label in 4-label schema")
    print(CSSRS_4["Label"].value_counts(normalize=True) * 100)
