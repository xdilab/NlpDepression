#!/usr/bin/env python3

from Libraries import *

def getXfromBestModelfromTrials(trials, x):
    valid_trial_list = [trial for trial in trials if STATUS_OK == trial['result']['status']]
    losses = [float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result'][x]


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


def mergeDicts(list_of_dicts):
    first_dict = list_of_dicts[0]
    for i in range(1, len(list_of_dicts)):
        for label, statistics in list_of_dicts[i].items():
            for statistic, value in statistics.items():
                first_dict[label][statistic] = value
    return(first_dict)

def onehotEncode(labels):
    onehot_encoded = list()
    for value in labels:
        encoded = [0 for _ in range(max(labels) + 1)]
        encoded[value] = 1
        onehot_encoded.append(encoded)
    new_labels = convert_to_tensor(onehot_encoded)
    return new_labels

def printOverallResults(outputPath, fileName, n_label, model_name, emb_type, max_length, boolDict, numCV, model_type,
                        stats, hyperparameters, execTime, whole_results, fold_results):
    if boolDict["CV"]:
        outputPath = os.path.join(outputPath, "CV", f"[{numCV} Folds]")
    elif boolDict["split"]:
        outputPath = os.path.join(outputPath, "Split")

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    if boolDict["SMOTE"]:
        weighting = "SMOTE"
    elif boolDict["weight"]:
        weighting = "Loss"
    else:
        weighting = "None"

    hours, minutes, seconds = str(execTime).split(":")
    results = pd.DataFrame({"Number of labels":n_label, "Embedding":model_name, "Max Sentence Length":max_length,
                            "Model":model_type, "Knowledge Infusion": boolDict["KI"], "Weighting":weighting,
                            "Parameter Tuning":boolDict["tuning"] ,"CV Folds":numCV,
                            "Macro Average":stats["accuracy"],"Precision":stats["macro avg"]["precision"],
                            "Recall":stats["macro avg"]["recall"],"F1-score":stats["macro avg"]["f1-score"],
                            "AUC":stats["auc"], "New False Positive Rate":stats["graded_fp"],
                            "New False Negative Rate":stats["graded_fn"], "Ordinal Error":stats["ordinal_err"],
                            "Execution Time":f"{hours}H{minutes}M", "random.seed":seed, "np seed":seed, "tf seed":seed,
                            "train_test split seed":split_random_seed, "SMOTE seed":SMOTE_random_seed,
                            "KFold seed":KFold_shuffle_random_seed}, index=[0])
    for i in range(n_label):
        results[f"Label {i} Accuracy"]  = stats[str(i)]["acc"]

    if boolDict["split"]:
        print(hyperparameters)
        results[f"Hyperparameters"] = str(sorted(list(hyperparameters.items()), key=lambda x: x[0][0]))
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

    if boolDict["CV"]:
        if numCV == 5:
            if n_label == 4:
                results = results[["QID", "Number of labels", "Embedding", "Max Sentence Length", "Model", "Knowledge Infusion",
                                   "Weighting", "Parameter Tuning", "CV Folds","Label 0 Accuracy", "Label 1 Accuracy","Label 2 Accuracy","Label 3 Accuracy",
                                   "Macro Average","Precision", "Recall","F1-score", "AUC", "New False Positive Rate","New False Negative Rate",
                               "Ordinal Error", "Fold 1 Hyperparameters", "Fold 2 Hyperparameters", "Fold 3 Hyperparameters",
                                   "Fold 4 Hyperparameters","Fold 5 Hyperparameters", "Execution Time", "random.seed", "np seed",
                                   "tf seed", "train_test split seed", "SMOTE seed", "KFold seed"]]
            elif n_label == 5:
                results = results[["QID", "Number of labels", "Embedding", "Max Sentence Length", "Model", "Knowledge Infusion",
                                   "Weighting", "Parameter Tuning", "CV Folds","Label 0 Accuracy", "Label 1 Accuracy", "Label 2 Accuracy", "Label 3 Accuracy",
                                   "Label 4 Accuracy","Macro Average", "Precision", "Recall", "F1-score", "AUC", "New False Positive Rate",
                               "New False Negative Rate", "Ordinal Error", "Fold 1 Hyperparameters", "Fold 2 Hyperparameters",
                                   "Fold 3 Hyperparameters","Fold 4 Hyperparameters","Fold 5 Hyperparameters", "Execution Time",
                                   "random.seed", "np seed", "tf seed", "train_test split seed", "SMOTE seed", "KFold seed"]]
    elif boolDict["split"]:
        if n_label == 4:
            results = results[
                ["QID", "Number of labels", "Embedding", "Max Sentence Length", "Model", "Knowledge Infusion",
                 "Weighting", "Parameter Tuning", "CV Folds", "Label 0 Accuracy", "Label 1 Accuracy", "Label 2 Accuracy",
                 "Label 3 Accuracy","Macro Average", "Precision", "Recall", "F1-score", "AUC", "New False Positive Rate",
                 "New False Negative Rate","Ordinal Error", "Hyperparameters", "Execution Time", "random.seed", "np seed",
                 "tf seed", "train_test split seed", "SMOTE seed", "KFold seed"]]
        elif n_label == 5:
            results = results[
                ["QID", "Number of labels", "Embedding", "Max Sentence Length", "Model", "Knowledge Infusion",
                 "Weighting", "Parameter Tuning", "CV Folds", "Label 0 Accuracy", "Label 1 Accuracy", "Label 2 Accuracy",
                 "Label 3 Accuracy","Label 4 Accuracy", "Macro Average", "Precision", "Recall", "F1-score", "AUC",
                 "New False Positive Rate",
                 "New False Negative Rate", "Ordinal Error", "Hyperparameters", "Execution Time",
                 "random.seed", "np seed", "tf seed", "train_test split seed", "SMOTE seed", "KFold seed"]]

    file_path = os.path.join(outputPath, fileName)
    results.to_csv(file_path, mode="a", index=False, header = not os.path.exists(file_path))


    if boolDict["split"]:
        actual_vs_pred = f"[{qid}] (No CV) {model_name}, Max length of {max_length},{' No' if boolDict['SMOTE'] == False else ''} SMOTE, {model_type}, " \
                           f"{'No' if boolDict['KI'] == False else 'with'} Knowledge Infusion, Actual_vs_Predicted"
        conf_matrix_name = f"[{qid}] (No CV) {model_name}, Max length of {max_length},{' No' if boolDict['SMOTE'] == False else ''} SMOTE, {model_type}, " \
                           f"{'No' if boolDict['KI'] == False else 'with'} Knowledge Infusion, Confusion Matrix"
    else:
        actual_vs_pred = f"[{qid}] {model_name}, Max length of {max_length},{' No' if boolDict['SMOTE'] == False else ''} SMOTE, {model_type}, " \
                           f"{'No' if boolDict['KI'] == False else 'with'} Knowledge Infusion, {numCV} Folds, Actual_vs_Predicted"
        conf_matrix_name = f"[{qid}] {model_name}, Max length of {max_length},{' No' if boolDict['SMOTE'] == False else ''} SMOTE, {model_type}, " \
                           f"{'No' if boolDict['KI'] == False else 'with'} Knowledge Infusion, {numCV} Folds, Confusion Matrix"
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

def printPredictions(y_test, y_pred, n_lab, outputPath):
    if n_lab == 4:
        df = pd.DataFrame({"Actual":y_test, "Pred Label":y_pred})
        df.to_csv(os.path.join(outputPath, "PredictedLabels (4-Label).csv"))
    elif n_lab == 5:
        df = pd.DataFrame({"Actual": y_test, "Pred Label": y_pred})
        df.to_csv(os.path.join(outputPath, "PredictedLabels (4-Label).csv"))

