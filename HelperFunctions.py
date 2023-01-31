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

def printMLMResults(outputPath, model_name, mask_strat, mlm_y_loss, mlm_params):
    # MLM training loss
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(111)
    ax1.plot(range(1, mlm_params["epochs"] + 1), mlm_y_loss["train"], color='red', marker='o',
             linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['acc_v'], legend='brief', label="accuracy")
    plt.xticks(range(1, mlm_params["epochs"] + 1))
    ax1.set_title(f'MLM training loss per epoch')  # , y=1.05, size=15)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(['train_loss'])
    fig.savefig(os.path.join(outputPath, f"[{model_name} with {mask_strat} masking] MLM training loss per epoch.png"))

def printOverallResults(outputPath, fileName, n_label, model_name, max_length, boolDict, numCV, model_type,
                        stats, hyperparameters, execTime, whole_results, fold_results, datasets, mask_strat, pretrain_bool,
                        mlm_params, mlm_y_loss, transferLearning, mainTaskBool, few_shot):
    """
    :param outputPath:
        path to folder to save all files
    :param fileName:
        name of file to save results (csv with overall results)
    :param n_label:
        For now, the number of CSSRS labels.
    :param model_name:
        Type of transformer model
    :param max_length:
        Maximum token length
    :param boolDict:
        Dictionary of boolean values: Tradiational train-test split, Cross-validation, Knowledge infusion,
                                      Class-based loss weighting, and parameter tuning
    :param numCV:
        Number of cross-validation folds
    :param model_type:
        Type of model on head of tranformer/embeddings. Note that 'transformer' only means not using CNN/GRU/LSTM head
    :param stats:
        Dict of various statistics
    :param hyperparameters:
        List of hyperparameters for each fold
    :param execTime:
        The total execution time of model
    :param whole_results:
        The overall results
    :param fold_results:
        Results for each fold
    :param datasets:
        The datasets utilized for pre-training and main task
    :param mask_strat:
        Type of masking strategy used for MLM. Note that only random masking currently supported
    :param pretrain_bool:
        Boolean value whether MLM pre-training performed
    :param mlm_params:
        Dict of hyperparameters for pre-training
    :return:
    """
    if boolDict["CV"]:
        outputPath = os.path.join(outputPath, "CV", f"[{numCV} Folds]")
    elif boolDict["split"]:
        outputPath = os.path.join(outputPath, "Split")

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    if boolDict["weight"]:
        weighting = "Loss"
    else:
        weighting = "None"

    if transferLearning:
        pretrain_str = "Transfer"
        mask_str = mask_strat
        pretrain_data = datasets["pretrain"]
        transf_params = str(mlm_params)
    else:
        pretrain_str = "Base"
        mask_str = ""
        pretrain_data = ""
        transf_params = ""

    if mainTaskBool:
        task_data = datasets["task"]
        macro_avg = stats["accuracy"]
        precision = stats["macro avg"]["precision"]
        recall = stats["macro avg"]["recall"]
        flScore = stats["macro avg"]["f1-score"]
        AUC = stats["auc"]
        nfp = stats["graded_fp"]
        nfn = stats["graded_fn"]
        ord_err = stats["ordinal_err"]
    else:
        task_data = ""
        macro_avg = ""
        precision = ""
        recall = ""
        flScore = ""
        AUC = ""
        nfp = ""
        nfn = ""
        ord_err = ""

    if few_shot["bool"]:
        few_shot_k = few_shot["k"]
    else:
        few_shot_k = ""

    hours, minutes, seconds = str(execTime).split(":")
    results = pd.DataFrame({"Number of labels":n_label, "Dynamic Model":model_name,"Pre-training":pretrain_str,
                            "Transfer Dataset":pretrain_data, "Masking":mask_str, "Transfer Hyperparameters":transf_params,
                            "Task Dataset":task_data,"Max Sentence Length":max_length,
                            "Model":model_type, "Weighting":weighting,
                            "Parameter Tuning":boolDict["tuning"], "Few-Shot Learning":few_shot["bool"], "K":few_shot_k,
                            "CV Folds":numCV, "Macro Average":macro_avg,"Precision":precision,
                            "Recall":recall,"F1-score":flScore,
                            "AUC":AUC, "New False Positive Rate":nfp,
                            "New False Negative Rate":nfn, "Ordinal Error":ord_err,
                            "Execution Time":f"{hours}H{minutes}M", "random.seed":seed, "np seed":seed, "tf seed":seed,
                            "train_test split seed":split_random_seed, "SMOTE seed":SMOTE_random_seed,
                            "KFold seed":KFold_shuffle_random_seed,"Notes":""}, index=[0])

    if mainTaskBool:
        for i in range(n_label):
            results[f"Label {i} Accuracy"] = stats[str(i)]["acc"]

        if boolDict["split"]:
            results[f"Hyperparameters"] = str(sorted(list(hyperparameters.items()), key=lambda x: x[0][0]))
        else:
            for i in range(numCV):
                results[f"Fold {i+1} Hyperparameters"] = str(sorted(list(hyperparameters[i].items()), key=lambda x: x[0][0]))
    else:
        for i in range(n_label):
            results[f"Label {i} Accuracy"] = ""

        if boolDict["split"]:
            results[f"Hyperparameters"] = ""
        else:
            for i in range(numCV):
                results[f"Fold {i+1} Hyperparameters"] = ""

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
                results = results[["QID", "Number of labels", "Dynamic Model", "Pre-training", "Transfer Dataset",
                                   "Masking", "Transfer Hyperparameters", "Task Dataset","Max Sentence Length", "Model",
                                   "Weighting", "Parameter Tuning", "Few-Shot Learning", "K", "CV Folds","Label 0 Accuracy", "Label 1 Accuracy","Label 2 Accuracy","Label 3 Accuracy",
                                   "Macro Average","Precision", "Recall","F1-score", "AUC", "New False Positive Rate","New False Negative Rate",
                               "Ordinal Error", "Fold 1 Hyperparameters", "Fold 2 Hyperparameters", "Fold 3 Hyperparameters",
                                   "Fold 4 Hyperparameters","Fold 5 Hyperparameters", "Execution Time", "random.seed", "np seed",
                                   "tf seed", "train_test split seed", "SMOTE seed", "KFold seed", "Notes"]]
            elif n_label == 5:
                results = results[["QID", "Number of labels", "Dynamic Model", "Pre-training", "Transfer Dataset",
                                   "Masking", "Transfer Hyperparameters", "Task Dataset","Max Sentence Length", "Model",
                                   "Weighting", "Parameter Tuning", "Few-Shot Learning", "K", "CV Folds","Label 0 Accuracy", "Label 1 Accuracy", "Label 2 Accuracy", "Label 3 Accuracy",
                                   "Label 4 Accuracy","Macro Average", "Precision", "Recall", "F1-score", "AUC", "New False Positive Rate",
                               "New False Negative Rate", "Ordinal Error", "Fold 1 Hyperparameters", "Fold 2 Hyperparameters",
                                   "Fold 3 Hyperparameters","Fold 4 Hyperparameters","Fold 5 Hyperparameters", "Execution Time",
                                   "random.seed", "np seed", "tf seed", "train_test split seed", "SMOTE seed", "KFold seed", "Notes"]]
    elif boolDict["split"]:
        if n_label == 4:
            results = results[
                ["QID", "Number of labels", "Dynamic Model", "Pre-training", "Transfer Dataset","Masking",
                 "Transfer Hyperparameters", "Task Dataset","Max Sentence Length", "Model",
                 "Weighting", "Parameter Tuning", "Few-Shot Learning", "K", "CV Folds", "Label 0 Accuracy", "Label 1 Accuracy", "Label 2 Accuracy",
                 "Label 3 Accuracy","Macro Average", "Precision", "Recall", "F1-score", "AUC", "New False Positive Rate",
                 "New False Negative Rate","Ordinal Error", "Hyperparameters", "Execution Time", "random.seed", "np seed",
                 "tf seed", "train_test split seed", "SMOTE seed", "KFold seed", "Notes"]]
        elif n_label == 5:
            results = results[
                ["QID", "Number of labels", "Dynamic Model", "Pre-training", "Transfer Dataset","Masking",
                 "Transfer Hyperparameters", "Task Dataset","Max Sentence Length", "Model",
                 "Weighting", "Parameter Tuning", "Few-Shot Learning", "K", "CV Folds", "Label 0 Accuracy", "Label 1 Accuracy", "Label 2 Accuracy",
                 "Label 3 Accuracy","Label 4 Accuracy", "Macro Average", "Precision", "Recall", "F1-score", "AUC",
                 "New False Positive Rate",
                 "New False Negative Rate", "Ordinal Error", "Hyperparameters", "Execution Time",
                 "random.seed", "np seed", "tf seed", "train_test split seed", "SMOTE seed", "KFold seed", "Notes"]]

    file_path = os.path.join(outputPath, fileName)
    results.to_csv(file_path, mode="a", index=False, header = not os.path.exists(file_path))


    if mainTaskBool:
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


    if pretrain_bool:
        # MLM training loss
        fig = plt.figure(figsize=(15, 15))
        ax1 = fig.add_subplot(111)
        ax1.plot(range(1, mlm_params["epochs"] + 1), mlm_y_loss["train"], color='red', marker='o',
                 linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['acc_v'], legend='brief', label="accuracy")
        plt.xticks(range(1, mlm_params["epochs"] + 1))
        ax1.set_title(f'MLM training loss per epoch')  # , y=1.05, size=15)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend(['train_loss'])
        fig.savefig(os.path.join(outputPath, f"[{qid}] MLM training loss per epoch.png"))

    if mainTaskBool:
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
            fig.savefig(os.path.join(outputPath, f"[{qid}] Fold  {i+1} - Train & Val Acc_Loss_AUC.png"))

        # plt.tight_layout()
        # plt.show()

def printOverallResultsMultiTask(outputPath, fileName, n_label, model_name, max_length, boolDict, numCV, model_type,
                        stats1, stats2, hyperparameters, execTime, whole_results, fold_results, datasets, mask_strat, pretrain_bool,
                        mlm_params, mlm_y_loss, transferLearning, mainTaskBool, few_shot):
    """
    :param outputPath:
        path to folder to save all files
    :param fileName:
        name of file to save results (csv with overall results)
    :param n_label:
        For now, the number of CSSRS labels.
    :param model_name:
        Type of transformer model
    :param max_length:
        Maximum token length
    :param boolDict:
        Dictionary of boolean values: Tradiational train-test split, Cross-validation, Knowledge infusion,
                                      Class-based loss weighting, and parameter tuning
    :param numCV:
        Number of cross-validation folds
    :param model_type:
        Type of model on head of tranformer/embeddings. Note that 'transformer' only means not using CNN/GRU/LSTM head
    :param stats:
        Dict of various statistics
    :param hyperparameters:
        List of hyperparameters for each fold
    :param execTime:
        The total execution time of model
    :param whole_results:
        The overall results
    :param fold_results:
        Results for each fold
    :param datasets:
        The datasets utilized for pre-training and main task
    :param mask_strat:
        Type of masking strategy used for MLM. Note that only random masking currently supported
    :param pretrain_bool:
        Boolean value whether MLM pre-training performed
    :param mlm_params:
        Dict of hyperparameters for pre-training
    :return:
    """
    if boolDict["CV"]:
        outputPath = os.path.join(outputPath, "CV", f"[{numCV} Folds]")
    elif boolDict["split"]:
        outputPath = os.path.join(outputPath, "Split")

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    if boolDict["weight"]:
        weighting = "Loss"
    else:
        weighting = "None"

    if transferLearning:
        pretrain_str = "Transfer"
        mask_str = mask_strat
        pretrain_data = datasets["pretrain"]
        transf_params = str(mlm_params)
    else:
        pretrain_str = "Base"
        mask_str = ""
        pretrain_data = ""
        transf_params = ""

    if mainTaskBool:
        task_data1 = datasets["multitask"][0]
        macro_avg1 = stats1["accuracy"]
        precision1 = stats1["macro avg"]["precision"]
        recall1 = stats1["macro avg"]["recall"]
        flScore1 = stats1["macro avg"]["f1-score"]
        AUC1 = stats1["auc"]
        nfp1 = stats1["graded_fp"]
        nfn1 = stats1["graded_fn"]
        ord_err1 = stats1["ordinal_err"]

        task_data2 = datasets["multitask"][1]
        macro_avg2 = stats2["accuracy"]
        precision2 = stats2["macro avg"]["precision"]
        recall2 = stats2["macro avg"]["recall"]
        flScore2 = stats2["macro avg"]["f1-score"]
        AUC2 = stats2["auc"]
        nfp2 = stats2["graded_fp"]
        nfn2 = stats2["graded_fn"]
        ord_err2 = stats2["ordinal_err"]

    else:
        task_data1 = ""
        macro_avg1 = ""
        precision1 = ""
        recall1 = ""
        flScore1 = ""
        AUC1 = ""
        nfp1 = ""
        nfn1 = ""
        ord_err1 = ""

        task_data2 = ""
        macro_avg2 = ""
        precision2 = ""
        recall2 = ""
        flScore2 = ""
        AUC2 = ""
        nfp2 = ""
        nfn2 = ""
        ord_err2 = ""

    if few_shot["bool"]:
        few_shot_k = few_shot["k"]
    else:
        few_shot_k = ""

    hours, minutes, seconds = str(execTime).split(":")
    results = pd.DataFrame({"Number of labels": n_label, "Dynamic Model": model_name, "Pre-training": pretrain_str,
                            "Transfer Dataset": pretrain_data, "Masking": mask_str,
                            "Transfer Hyperparameters": transf_params,
                            "Task Dataset1": task_data1, "Task Dataset2": task_data2, "Max Sentence Length": max_length,
                            "Model": model_type, "Weighting": weighting, "Parameter Tuning": boolDict["tuning"],
                            "Few-Shot Learning": few_shot["bool"], "K": few_shot_k,
                            "CV Folds": numCV,
                            f"Macro Average CSSRS": macro_avg1, f"AUC CSSRS": AUC1,
                            "Dividing Line": "",
                            f"Macro Average UMD": macro_avg2, f"AUC UMD": AUC2,
                            "Execution Time": f"{hours}H{minutes}M", "random.seed": seed, "np seed": seed,
                            "tf seed": seed,
                            "train_test split seed": split_random_seed, "SMOTE seed": SMOTE_random_seed,
                            "KFold seed": KFold_shuffle_random_seed, "Notes": ""}, index=[0])

    # results = pd.DataFrame({"Number of labels":n_label, "Dynamic Model":model_name,"Pre-training":pretrain_str,
    #                         "Transfer Dataset":pretrain_data, "Masking":mask_str, "Transfer Hyperparameters":transf_params,
    #                         "Task Dataset1":task_data1, "Task Dataset2":task_data2,"Max Sentence Length":max_length,
    #                         "Model":model_type, "Weighting":weighting, "Parameter Tuning":boolDict["tuning"],
    #                         "Few-Shot Learning":few_shot["bool"], "K":few_shot_k,
    #                         "CV Folds":numCV,
    #                         f"Macro Average {task_data1}":macro_avg1,f"Precision {task_data1}":precision1,
    #                         f"Recall {task_data1}":recall1,f"F1-score {task_data1}":flScore1,
    #                         f"AUC {task_data1}":AUC1, f"New False Positive Rate {task_data1}":nfp1,
    #                         f"New False Negative Rate {task_data1}":nfn1, f"Ordinal Error {task_data1}":ord_err1,
    #                         "Dividing Line": "",
    #                         f"Macro Average {task_data2}": macro_avg2, f"Precision {task_data2}": precision2,
    #                         f"Recall {task_data1}": recall2, f"F1-score {task_data2}": flScore2,
    #                         f"AUC {task_data2}": AUC2, f"New False Positive Rate {task_data2}": nfp2,
    #                         f"New False Negative Rate {task_data2}": nfn2, f"Ordinal Error {task_data2}": ord_err2,
    #                         "Execution Time":f"{hours}H{minutes}M", "random.seed":seed, "np seed":seed, "tf seed":seed,
    #                         "train_test split seed":split_random_seed, "SMOTE seed":SMOTE_random_seed,
    #                         "KFold seed":KFold_shuffle_random_seed,"Notes":""}, index=[0])

    if mainTaskBool:
        for i in range(n_label):
            results[f"CSSRS Label {i} Accuracy"] = stats1[str(i)]["acc"]
        for i in range(n_label):
            results[f"UMD Label {i} Accuracy"] = stats2[str(i)]["acc"]

        if boolDict["split"]:
            results[f"Hyperparameters"] = str(sorted(list(hyperparameters.items()), key=lambda x: x[0][0]))
        else:
            for i in range(numCV):
                results[f"Fold {i+1} Hyperparameters"] = str(sorted(list(hyperparameters[i].items()), key=lambda x: x[0][0]))
    else:
        for i in range(n_label):
            results[f"{task_data1} Label {i} Accuracy"] = ""
        for i in range(n_label):
            results[f"{task_data2} Label {i} Accuracy"] = ""

        if boolDict["split"]:
            results[f"Hyperparameters"] = ""
        else:
            for i in range(numCV):
                results[f"Fold {i+1} Hyperparameters"] = ""

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
                results = results[["QID", "Number of labels", "Dynamic Model", "Pre-training", "Transfer Dataset",
                                   "Masking", "Transfer Hyperparameters", "Task Dataset1", "Task Dataset2","Max Sentence Length", "Model",
                                   "Weighting", "Parameter Tuning", "Few-Shot Learning", "K", "CV Folds",
                                   "CSSRS Label 0 Accuracy", "CSSRS Label 1 Accuracy","CSSRS Label 2 Accuracy","CSSRS Label 3 Accuracy",
                                   "Macro Average CSSRS", "AUC CSSRS",
                                   "Dividing Line",
                                   "UMD Label 0 Accuracy", "UMD Label 1 Accuracy",
                                   "UMD Label 2 Accuracy", "UMD Label 3 Accuracy",
                                   "Macro Average UMD", "AUC UMD",
                                   "Fold 1 Hyperparameters", "Fold 2 Hyperparameters", "Fold 3 Hyperparameters",
                                   "Fold 4 Hyperparameters","Fold 5 Hyperparameters", "Execution Time", "random.seed", "np seed",
                                   "tf seed", "train_test split seed", "SMOTE seed", "KFold seed", "Notes"]]
        elif numCV == 2:
            results = results[["QID", "Number of labels", "Dynamic Model", "Pre-training", "Transfer Dataset",
                               "Masking", "Transfer Hyperparameters", "Task Dataset1", "Task Dataset2",
                               "Max Sentence Length", "Model",
                               "Weighting", "Parameter Tuning", "Few-Shot Learning", "K", "CV Folds",
                               "CSSRS Label 0 Accuracy", "CSSRS Label 1 Accuracy", "CSSRS Label 2 Accuracy",
                               "CSSRS Label 3 Accuracy",
                               "Macro Average CSSRS", "AUC CSSRS",
                               "Dividing Line",
                               "UMD Label 0 Accuracy", "UMD Label 1 Accuracy",
                               "UMD Label 2 Accuracy", "UMD Label 3 Accuracy",
                               "Macro Average UMD", "AUC UMD",
                               "Fold 1 Hyperparameters", "Fold 2 Hyperparameters", "Execution Time", "random.seed",
                               "np seed", "tf seed", "train_test split seed", "SMOTE seed", "KFold seed", "Notes"]]
            # elif n_label == 5:
            #     results = results[["QID", "Number of labels", "Dynamic Model", "Pre-training", "Transfer Dataset",
            #                        "Masking", "Transfer Hyperparameters", "Task Dataset","Max Sentence Length", "Model",
            #                        "Weighting", "Parameter Tuning", "Few-Shot Learning", "K", "CV Folds","Label 0 Accuracy", "Label 1 Accuracy", "Label 2 Accuracy", "Label 3 Accuracy",
            #                        "Label 4 Accuracy","Macro Average", "Precision", "Recall", "F1-score", "AUC", "New False Positive Rate",
            #                    "New False Negative Rate", "Ordinal Error", "Fold 1 Hyperparameters", "Fold 2 Hyperparameters",
            #                        "Fold 3 Hyperparameters","Fold 4 Hyperparameters","Fold 5 Hyperparameters", "Execution Time",
            #                        "random.seed", "np seed", "tf seed", "train_test split seed", "SMOTE seed", "KFold seed", "Notes"]]
    # elif boolDict["split"]:
    #     if n_label == 4:
    #         results = results[
    #             ["QID", "Number of labels", "Dynamic Model", "Pre-training", "Transfer Dataset","Masking",
    #              "Transfer Hyperparameters", "Task Dataset","Max Sentence Length", "Model",
    #              "Weighting", "Parameter Tuning", "Few-Shot Learning", "K", "CV Folds", "Label 0 Accuracy", "Label 1 Accuracy", "Label 2 Accuracy",
    #              "Label 3 Accuracy","Macro Average", "Precision", "Recall", "F1-score", "AUC", "New False Positive Rate",
    #              "New False Negative Rate","Ordinal Error", "Hyperparameters", "Execution Time", "random.seed", "np seed",
    #              "tf seed", "train_test split seed", "SMOTE seed", "KFold seed", "Notes"]]
    #     elif n_label == 5:
    #         results = results[
    #             ["QID", "Number of labels", "Dynamic Model", "Pre-training", "Transfer Dataset","Masking",
    #              "Transfer Hyperparameters", "Task Dataset","Max Sentence Length", "Model",
    #              "Weighting", "Parameter Tuning", "Few-Shot Learning", "K", "CV Folds", "Label 0 Accuracy", "Label 1 Accuracy", "Label 2 Accuracy",
    #              "Label 3 Accuracy","Label 4 Accuracy", "Macro Average", "Precision", "Recall", "F1-score", "AUC",
    #              "New False Positive Rate",
    #              "New False Negative Rate", "Ordinal Error", "Hyperparameters", "Execution Time",
    #              "random.seed", "np seed", "tf seed", "train_test split seed", "SMOTE seed", "KFold seed", "Notes"]]

    file_path = os.path.join(outputPath, fileName)
    results.to_csv(file_path, mode="a", index=False, header = not os.path.exists(file_path))


    if mainTaskBool:
        if boolDict["split"]:
            actual_vs_pred = f"[{qid}] (No CV) {model_name}, Max length of {max_length},{' No' if boolDict['SMOTE'] == False else ''} SMOTE, {model_type}, " \
                               f"{'No' if boolDict['KI'] == False else 'with'} Knowledge Infusion, Actual_vs_Predicted"
            conf_matrix_name = f"[{qid}] (No CV) {model_name}, Max length of {max_length},{' No' if boolDict['SMOTE'] == False else ''} SMOTE, {model_type}, " \
                               f"{'No' if boolDict['KI'] == False else 'with'} Knowledge Infusion, Confusion Matrix"
        else:
            actual_vs_pred1 = f"[{qid}] {task_data1} {model_name}, Max length of {max_length},{' No' if boolDict['SMOTE'] == False else ''} SMOTE, {model_type}, " \
                               f"{numCV} Folds, Actual_vs_Predicted"
            conf_matrix_name1 = f"[{qid}] {task_data1} {model_name}, Max length of {max_length},{' No' if boolDict['SMOTE'] == False else ''} SMOTE, {model_type}, " \
                               f"{numCV} Folds, Confusion Matrix"
            actual_vs_pred2 = f"[{qid}] {task_data2} {model_name}, Max length of {max_length},{' No' if boolDict['SMOTE'] == False else ''} SMOTE, {model_type}, " \
                             f"{numCV} Folds, Actual_vs_Predicted"
            conf_matrix_name2 = f"[{qid}] {task_data2} {model_name}, Max length of {max_length},{' No' if boolDict['SMOTE'] == False else ''} SMOTE, {model_type}, " \
                               f"{numCV} Folds, Confusion Matrix"
        folder = os.listdir(outputPath)

        #Check if file with same name exists
        #If so, add the appropriate number at the end to designate different results using same parameters
        file_count1 = 0
        file_count2 = 0
        for file in folder:
            if conf_matrix_name1 in file:
                file_count1 = file_count1 + 1
            if conf_matrix_name2 in file:
                file_count2 = file_count2 + 1

        matDF1 = pd.DataFrame(stats1["matrix"], index=[i for i in range(n_label)], columns=[i for i in range(n_label)])
        ax1 = sns.heatmap(matDF1, annot=True, cmap="Blues", fmt='d').get_figure()

        if file_count1 == 0:
            ax1.savefig(os.path.join(outputPath, conf_matrix_name1 + f".png"))
            whole_results.to_csv(os.path.join(outputPath, actual_vs_pred1 + ".csv"), index=False)
        else:
            ax1.savefig(os.path.join(outputPath,conf_matrix_name1 + f" ({file_count1}).png"))
            whole_results.to_csv(os.path.join(outputPath, actual_vs_pred1 + f" ({file_count1}).csv"), index=False)

        plt.clf()

        matDF2 = pd.DataFrame(stats2["matrix"], index=[i for i in range(n_label)], columns=[i for i in range(n_label)])
        ax2 = sns.heatmap(matDF2, annot=True, cmap="Blues", fmt='d').get_figure()

        if file_count2 == 0:
            ax2.savefig(os.path.join(outputPath, conf_matrix_name2 + f".png"))
        else:
            ax2.savefig(os.path.join(outputPath, conf_matrix_name2 + f" ({file_count2}).png"))

        plt.clf()

    if pretrain_bool:
        # MLM training loss
        fig = plt.figure(figsize=(15, 15))
        ax1 = fig.add_subplot(111)
        ax1.plot(range(1, mlm_params["epochs"] + 1), mlm_y_loss["train"], color='red', marker='o',
                 linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['acc_v'], legend='brief', label="accuracy")
        plt.xticks(range(1, mlm_params["epochs"] + 1))
        ax1.set_title(f'MLM training loss per epoch')  # , y=1.05, size=15)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend(['train_loss'])
        fig.savefig(os.path.join(outputPath, f"[{qid}] MLM training loss per epoch.png"))

    if mainTaskBool:
        # Fold training/validation statistics
        for i in range(len(fold_results)):
            fig = plt.figure(figsize=(15, 15))
            ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=1, colspan=1)
            ax1 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, colspan=1)
            ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, colspan=1)

            # Plotting Loss
            ax0.plot(fold_results[i]['epochs'], fold_results[i]['train_CSSRS_loss'], color='red', marker='o',
                     linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['acc_v'], legend='brief', label="accuracy")
            ax0.plot(fold_results[i]['epochs'], fold_results[i]['val_CSSRS_loss'], color='green', marker='o',
                     linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['val_acc'], legend='brief', label="val_accuracy")
            ax0.plot(fold_results[i]['epochs'], fold_results[i]['train_UMD_loss'], color='black', marker='o',
                     linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['loss_v'], legend='brief', label="loss")
            ax0.plot(fold_results[i]['epochs'], fold_results[i]['val_UMD_loss'], color='blue', marker='o',
                     linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['val_loss'], legend='brief', label="val_loss")
            ax0.set_title(f'Fold {i+1}: Training and validation loss')  # , y=1.05, size=15)
            ax0.legend(['train_CSSRS_loss', 'val_CSSRS_loss', 'train_UMD_loss', 'val_UMD_loss'])

            # Plotting Accuracy
            ax1.plot(fold_results[i]['epochs'], fold_results[i]['train_CSSRS_acc'], color='red', marker='o',
                     linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['acc_v'], legend='brief', label="accuracy")
            ax1.plot(fold_results[i]['epochs'], fold_results[i]['val_CSSRS_acc'], color='green', marker='o',
                     linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['val_acc'], legend='brief', label="val_accuracy")
            ax1.plot(fold_results[i]['epochs'], fold_results[i]['train_UMD_acc'], color='black', marker='o',
                     linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['loss_v'], legend='brief', label="loss")
            ax1.plot(fold_results[i]['epochs'], fold_results[i]['val_UMD_acc'], color='blue', marker='o',
                     linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['val_loss'], legend='brief', label="val_loss")
            ax1.set_title(f'Fold {i + 1}: Training and validation accuracy')  # , y=1.05, size=15)
            ax1.legend(['train_CSSRS_acc', 'val_CSSRS_acc', 'train_UMD_acc', 'val_UMD_acc'])

            # Plotting AUC
            ax2.set_title(f'Fold {i+1}: Training and validation AUC')
            ax2.plot(fold_results[i]['epochs'], fold_results[i]['train_CSSRS_auc'], color='red', marker='o',
                     linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['acc_v'], legend='brief', label="accuracy")
            ax2.plot(fold_results[i]['epochs'], fold_results[i]['val_CSSRS_auc'], color='green', marker='o',
                     linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['val_acc'], legend='brief', label="val_accuracy")
            ax2.plot(fold_results[i]['epochs'], fold_results[i]['train_UMD_auc'], color='black', marker='o',
                     linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['acc_v'], legend='brief', label="accuracy")
            ax2.plot(fold_results[i]['epochs'], fold_results[i]['val_UMD_auc'], color='blue', marker='o',
                     linestyle='dashed')  # sns.lineplot(ax=ax0,x=final_results['epochs'],y=final_results['val_acc'], legend='brief', label="val_accuracy")
            ax2.set_title(f'Fold {i + 1}: Training and validation AUC')
            ax2.legend(['train_CSSRS_auc', 'val_CSSRS_auc', 'train_UMD_auc', 'val_UMD_auc'])
            fig.savefig(os.path.join(outputPath, f"[{qid}] Fold  {i+1} - Train & Val Acc_Loss_AUC.png"))

        # plt.tight_layout()
        # plt.show()

def getLabels(df, num_labels, dataset):
    if dataset == "CSSRS":

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
            return (df, inv_map)
    elif dataset == "UMD":
        label_conversion = {"a": 0,
                            "b": 1,
                            "c": 2,
                            "d": 3}

        df = df.replace({"Label": label_conversion})
        inv_map = {val: key for key, val in label_conversion.items()}
        return (df, inv_map)

def printPredictions(y_test, y_pred, n_lab, outputPath):
    if n_lab == 4:
        df = pd.DataFrame({"Actual":y_test, "Pred Label":y_pred})
        df.to_csv(os.path.join(outputPath, "PredictedLabels (4-Label).csv"))
    elif n_lab == 5:
        df = pd.DataFrame({"Actual": y_test, "Pred Label": y_pred})
        df.to_csv(os.path.join(outputPath, "PredictedLabels (4-Label).csv"))

