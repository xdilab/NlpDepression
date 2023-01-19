#!/usr/bin/env python3

from Libraries import *
from HelperFunctions import mergeDicts


def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()

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

def getTokenStats(df, len_of_interest = 512):
    token_lengths = []
    for i in range(len(df["model_input"])):
        token_lengths.append(len(df["model_input"][i]["input_ids"][0]))
    print(f"Max Length: {max(token_lengths)}")
    print(f"Min Length: {min(token_lengths)}")
    print(f"Mean Length: {mean(token_lengths)}")
    print(f"Median Length: {median(token_lengths)}")

    num_under = 0
    for tok_length in token_lengths:
        if tok_length <= 512:
            num_under += 1
    print(f"Percent under or exactly {len_of_interest} tokens: {num_under/len(token_lengths)}")