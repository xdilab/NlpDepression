import keras.optimizers
from Libraries import *
from ModelFunctions import denseNN, cnnModel, cnnModel2, objectiveFunctionCNN, RNNModel, objectiveFunctionRNN, \
TFSentenceTransformer, E2ESentenceTransformer, multiTaskModel, multiTaskModel1
from HelperFunctions import getLabels, extractList,getXfromBestModelfromTrials, printPredictions, \
    printOverallResults, onehotEncode, printMLMResults, printOverallResultsMultiTask
from EmbeddingFunctions import BERT_embeddings, getTokens, customDataset, runModel, MaskingFunction, getFeatures
from StatisticsFunctions import getStatistics, getSummStats, softmax
from ImportFunctions import importUMD, importCSSRS, getModel, getTokenizer, getRegularModel, getMainTaskModel, getMultiTaskModel

def MultiTaskrunFold(outputPath, filespath, model, model_name, tokenizer, modelType, max_length, num_labels,
            boolDict, hyperparameters, n_folds, fold_num, datasets, mask_strat, X_train_fold1, X_train_fold2,
                     y_train_fold1, y_train_fold2, X_val_fold1, X_val_fold2, y_val_fold1, y_val_fold2,
                     X_test_fold1, X_test_fold2, y_test_fold1, y_test_fold2, mlm_pretrain_bool, mlm_params, few_shot):

    # Convert to BERT embeddings for BERT models
    if modelType != "transformer":
        # text, tokenizer, model, 2d or 3d embeddings
        X_train_emb_fold = getFeatures(text=X_train_fold, tokenizer=tokenizer, model=model, model_name=model_name,
                                       max_length=max_length, return_tensor_type="pt")

        X_test_emb_fold = getFeatures(text=X_test_fold, tokenizer=tokenizer, model=model, model_name=model_name,
                                      max_length=max_length, return_tensor_type="pt")
        number_channels, number_features = X_train_emb_fold.shape[1], X_train_emb_fold.shape[2]


    if modelType != "transformer":
        modelTrain = tf.convert_to_tensor(X_train_emb_fold)
        modelTest = tf.convert_to_tensor(X_test_emb_fold)
    else:
        input_ids, input_masks, input_segments = [], [], []
        input_ids2, input_masks2, input_segments2 = [], [], []

        trainOut1 = tokenizer(X_train_fold1.to_list(), return_tensors="np",max_length=max_length,
                                                truncation=True,padding='max_length')
        trainOut2 = tokenizer(X_train_fold2.to_list(), return_tensors="np", max_length=max_length,
                             truncation=True, padding='max_length')
        if type(X_val_fold1)==pd.Series:
            valOut1 = tokenizer(X_val_fold1.to_list(), return_tensors="np", max_length=max_length,
                                 truncation=True, padding='max_length')
            ValOut2 = tokenizer(X_val_fold2.to_list(), return_tensors="np", max_length=max_length,
                              truncation=True, padding='max_length')

        testOut1 = tokenizer(X_test_fold1.to_list(), return_tensors="np", max_length=max_length,
                            truncation=True, padding='max_length')
        testOut2 = tokenizer(X_test_fold2.to_list(), return_tensors="np", max_length=max_length,
                            truncation=True, padding='max_length')

        # print(trainOut)
        # input("WAIT")

        if model_name != "ROBERTA" and model_name.upper() != "SBERT":
            modelTrain1 = {"input_ids":trainOut1['input_ids'],
                          "token_type_ids":trainOut1['token_type_ids'],
                          'attention_mask':trainOut1['attention_mask']}
            modelTrain2 = {"input_ids": trainOut2['input_ids'],
                          "token_type_ids": trainOut2['token_type_ids'],
                          'attention_mask': trainOut2['attention_mask']}
            if type(X_val_fold1)==pd.Series:
                modelVal1 = {"input_ids": valOut1['input_ids'],
                              "token_type_ids": valOut1['token_type_ids'],
                              'attention_mask': valOut1['attention_mask']}
                modelVal2 = {"input_ids": valOut2['input_ids'],
                            "token_type_ids": valOut2['token_type_ids'],
                            'attention_mask': valOut2['attention_mask']}

            modelTest1 = {"input_ids": testOut1['input_ids'],
                          "token_type_ids": testOut1['token_type_ids'],
                          'attention_mask': testOut1['attention_mask']}
            modelTest2 = {"input_ids": testOut2['input_ids'],
                         "token_type_ids": testOut2['token_type_ids'],
                         'attention_mask': testOut2['attention_mask']}
        else:
            modelTrain1 = {"input_ids": trainOut1['input_ids'],
                          'attention_mask': trainOut1['attention_mask']}
            modelTrain2 = {"input_ids": trainOut2['input_ids'],
                          'attention_mask': trainOut2['attention_mask']}
            if type(X_val_fold1)==pd.Series:
                modelVal1 = {"input_ids": valOut1['input_ids'],
                              'attention_mask': valOut1['attention_mask']}
                modelVal2 = {"input_ids": valOut2['input_ids'],
                            'attention_mask': valOut2['attention_mask']}

            modelTest1 = {"input_ids": testOut1['input_ids'],
                         'attention_mask': testOut1['attention_mask']}
            modelTest2 = {"input_ids": testOut2['input_ids'],
                         'attention_mask': testOut2['attention_mask']}

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
        onehot1 = pd.get_dummies(pd.Series(y_train_fold1), drop_first=False)
        y_train_fold1 = convert_to_tensor(onehot1)
        onehotTest1 = pd.get_dummies(pd.Series(y_test_fold1.numpy()), drop_first=False)
        y_test_fold1 = convert_to_tensor(onehotTest1)

        onehot2 = pd.get_dummies(pd.Series(y_train_fold2), drop_first=False)
        y_train_fold2 = convert_to_tensor(onehot2)
        onehotTest2 = pd.get_dummies(pd.Series(y_test_fold2.numpy()), drop_first=False)
        y_test_fold2 = convert_to_tensor(onehotTest2)

    if type(X_val_fold1) == pd.Series:
        onehotVal1 = pd.get_dummies(pd.Series(y_val_fold1.numpy()), drop_first=False)
        y_val_fold1 = convert_to_tensor(onehotVal1)

        onehotVal2 = pd.get_dummies(pd.Series(y_val_fold2.numpy()), drop_first=False)
        y_val_fold2 = convert_to_tensor(onehotVal2)

    checkpointName = f"{model_name}_{modelType}_{'with' if boolDict['KI'] else 'no'}_KI_best_model.ckpt"


    es = EarlyStopping(monitor='val_auc', mode="max", patience=5, min_delta=0)
    # mc = ModelCheckpoint(checkpointName, monitor='val_auc', mode='max', verbose=0,
    #                      save_best_only=True)

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
    elif modelType == "transformer":

        # print(modelTrain1["input_ids"])
        # print(modelTrain1["input_ids"][0])
        # print(modelTrain1["input_ids"][0][0])
        # print(type(modelTrain1["input_ids"][0][0]))
        # input("WAIT")

        nnModel1 = multiTaskModel1(model, "CSSRS", max_length, num_labels)
        nnModel2 = multiTaskModel1(model, "UMD", max_length, num_labels)

        metrics = [tf.keras.metrics.CategoricalAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')]
        learn_rate = hyperparameters["learning_rate"]
        # learn_rate = tf.keras.optimizers.schedules.ExponentialDecay(hyperparameters["learning_rate"], decay_steps=30, decay_rate=0.95, staircase=True)

        if model_name.upper() == "SBERT":
            loss = {'CSSRS_Output': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    'UMD_Output': tf.keras.losses.CategoricalCrossentropy(from_logits=True)}
        else:
            loss = {'CSSRS_Output': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    'UMD_Output': tf.keras.losses.CategoricalCrossentropy(from_logits=False)}
            # loss = {'CSSRS_Output': tf.keras.losses.CategoricalCrossentropy(from_logits=False)}


        # nnModel.compile(optimizer=Adam(learning_rate=learn_rate),
        #              loss=loss,
        #              metrics=metrics)

        # pred = nnModel(modelTrain)
        # print(f"output shape: {pred.shape}")
        # nnModel([modelTrain]).summary()

    # if model_name.upper() != "SBERT":
    if boolDict["weight"]:
        if type(X_val_fold)==pd.Series:
            history = nnModel.fit(modelTrain, y_train_fold,
                                 validation_data=(modelVal, y_val_fold),
                                 epochs=hyperparameters["epochs"],
                                 batch_size=hyperparameters["batch_size"], callbacks=[es],
                                 class_weight=class_weights, verbose=2)
        else:
            history = nnModel.fit(modelTrain, y_train_fold,
                                  validation_data=(modelTrain, y_train_fold),
                                  epochs=hyperparameters["epochs"],
                                  batch_size=hyperparameters["batch_size"], callbacks=[es],
                                  class_weight=class_weights,verbose=2)
    else:
        if type(X_val_fold1)==pd.Series:
            history = nnModel.fit(modelTrain, y_train_fold,
                                 validation_data=(modelVal, y_val_fold),
                                 epochs=hyperparameters["epochs"],
                                 batch_size=hyperparameters["batch_size"], callbacks=[es],
                                  verbose=2)
        else:
            # print(modelTrain1)
            modelTrain1_dataset = tf.data.Dataset.from_tensor_slices((modelTrain1, y_train_fold1))
            modelTrain2_dataset = tf.data.Dataset.from_tensor_slices((modelTrain2, y_train_fold2))
            # print(modelTrain1_dataset)
            optimizer = keras.optimizers.Adam(learning_rate=hyperparameters["learning_rate"])
            loss_fn1 = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            loss_fn2 = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            train_accuracy1 = tf.keras.metrics.CategoricalAccuracy()
            train_accuracy2 = tf.keras.metrics.CategoricalAccuracy()

            num_epochs = hyperparameters["epochs"]
            batch_size = hyperparameters["batch_size"]

            modelTrain1_dataset = modelTrain1_dataset.batch(batch_size)
            modelTrain2_dataset = modelTrain2_dataset.batch(batch_size)

            loss_values1 = []
            acc_values1 = []

            loss_values2 = []
            acc_values2 = []

            for epoch in range(num_epochs):
                print("\nStart of epoch %d" % (epoch,))
                # for (step, (x_batch_train1, y_batch_train1)), (step2, (x_batch_train2, y_batch_train2)) in zip_longest(enumerate(modelTrain1_dataset), enumerate(modelTrain2_dataset), fillvalue= ""):
                batch_loss = 0
                for step, (x_batch_train1, y_batch_train1) in enumerate(modelTrain1_dataset):

                    with tf.GradientTape() as tape:
                        logits = nnModel1(x_batch_train1, training=True)
                        loss_value = loss_fn1(y_batch_train1, logits)
                        batch_loss += loss_value

                    grads = tape.gradient(loss_value, nnModel1.trainable_weights)
                    optimizer.apply_gradients(zip(grads, nnModel1.trainable_weights))
                    train_accuracy1.update_state(y_batch_train1, logits)


                loss_values1.append(batch_loss.numpy())

                # for step2, (x_batch_train2, y_batch_train2) in enumerate(modelTrain2_dataset):


                train_acc = train_accuracy1.result()
                acc_values1.append(train_acc)
                print("Training acc over epoch: %.4f" % (float(train_acc),))

                batch_loss = 0
                for step2, (x_batch_train2, y_batch_train2) in enumerate(modelTrain2_dataset):

                    with tf.GradientTape() as tape:
                        logits = nnModel2(x_batch_train2, training=True)
                        loss_value = loss_fn2(y_batch_train2, logits)
                        batch_loss += loss_value

                    grads = tape.gradient(loss_value, nnModel2.trainable_weights)
                    optimizer.apply_gradients(zip(grads, nnModel2.trainable_weights))
                    train_accuracy2.update_state(y_batch_train2, logits)

                loss_values2.append(batch_loss.numpy())

                # for step2, (x_batch_train2, y_batch_train2) in enumerate(modelTrain2_dataset):

                train_acc = train_accuracy2.result().numpy()
                acc_values2.append(train_acc)
                print("Training acc over epoch: %.4f" % (float(train_acc),))


            print(loss_values1)
            plt.plot(range(num_epochs), loss_values1)
            plt.xlabel("Epoch")
            plt.ylabel('Loss')
            plt.show()

            print(acc_values1)
            plt.plot(range(num_epochs), acc_values1)
            plt.xlabel("Epoch")
            plt.ylabel('Accuracy')
            plt.show()

            print(loss_values2)
            plt.plot(range(num_epochs), loss_values2)
            plt.xlabel("Epoch")
            plt.ylabel('Loss')
            plt.show()

            print(acc_values2)
            plt.plot(range(num_epochs), acc_values2)
            plt.xlabel("Epoch")
            plt.ylabel('Accuracy')
            plt.show()
            exit()

            # history = nnModel.fit([modelTrain1, modelTrain2],
            #                       {"CSSRS_Output": y_train_fold1, "UMD_Output": y_train_fold2},
            #                       validation_data=([modelTrain1,modelTrain2],
            #                                        {"CSSRS_Output": y_train_fold1, "UMD_Output": y_train_fold2}),
            #                       epochs=hyperparameters["epochs"],
            #                       batch_size=hyperparameters["batch_size"], callbacks=[es],
            #                       verbose=2)


    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    # nnModel = load_model(checkpointName)
    # scores = {}
    scores = nnModel.evaluate([modelTest1, modelTest2], {"CSSRS_Output": y_test_fold1, "UMD_Output": y_test_fold2}, verbose=0)
    y_pred_proba = nnModel.predict([modelTest1, modelTest2])


    return nnModel, history, scores, y_pred_proba, hyperparameters

def MultiTaskrun(outputPath, UMD_path, CSSRS_path, model_name, mlm_params, mlm_pretrain, transferLearning, mainTask, CSSRS_n_label,
        boolDict, hyperparameters, n_folds, modelType, max_length, datasets, mask_strat, few_shot):

    ##-----MLM Pre-training-----
    if mlm_pretrain:
        if not transferLearning:
            startTime = datetime.now()
        tokenizer, model = getModel(model_name)

        split_dataset_name = datasets["pretrain"].split("-")
        if split_dataset_name[0] == "UMD":
            if split_dataset_name[1] == "crowd":
                task_train, task_test = importUMD(UMD_path, split_dataset_name[1], split_dataset_name[2])
            elif split_dataset_name[1] == "expert":
                task_train, task_test = importUMD(UMD_path, split_dataset_name[1])

        elif split_dataset_name[0] == "CSSRS":
            CSSRS = importCSSRS(CSSRS_path, num_labels=CSSRS_n_label)
            # Make all text lowercase
            text = CSSRS["Post"].apply(lambda x: x.lower())
            labels = CSSRS["Label"]
        else:
            print("Incorrect input for pretrain dataset. Exiting...")
            exit(1)
        # For now, remove all posts with empty post_body (i.e. just post title)
        task_train = task_train[task_train["post_body"].notnull()]  # Removes 72 posts
        task_test = task_test[task_test["post_body"].notnull()]  # Removes 5 posts
        # Sort user_id and timestamp in descending order for both
        task_train = task_train.sort_values(by=["user_id", "timestamp"], ascending=[True, True], ignore_index=True)
        task_test = task_test.sort_values(by=["user_id", "timestamp"], ascending=[True, True], ignore_index=True)
        task_train["model_input"] = task_train["post_body"].map(lambda x: MaskingFunction(x, tokenizer=tokenizer,
                                                                                          masking_strat=mask_strat,
                                                                                          custom_masking=False,
                                                                                          max_length=max_length,
                                                                                          return_tensor_type="pt"))
        task_test["model_input"] = task_test["post_body"].map(lambda x: MaskingFunction(x, tokenizer=tokenizer,
                                                                                        masking_strat=mask_strat,
                                                                                        custom_masking=False,
                                                                                        max_length=max_length,
                                                                                        return_tensor_type="pt"))

        train_model_input = {key: torch.stack([i[key][0] for i in task_train["model_input"]]) for key in
                             task_train["model_input"][0]}
        test_model_input = {key: torch.stack([i[key][0] for i in task_test["model_input"]]) for key in
                            task_test["model_input"][0]}

        # train_model_input2 = {}
        # train_model_input2["input_ids"] = train_model_input["input_ids"][:20]
        # train_model_input2["token_type_ids"] = train_model_input["token_type_ids"][:20]
        # train_model_input2["attention_mask"] = train_model_input["attention_mask"][:20]
        # train_model_input2["labels"] = train_model_input["labels"][:20]

        trainDataset = customDataset(train_model_input)
        testDataset = customDataset(test_model_input)

        ## From https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # and move our model over to the selected device
        model.to(device)
        # initialize optimizer
        optim = torch.optim.AdamW(model.parameters(), lr=mlm_params["learning_rate"])

        # activate training mode
        model.train()
        loader = torch.utils.data.DataLoader(trainDataset, batch_size=mlm_params["batch_size"], shuffle=True)

        mlm_y_loss = {}  # loss history
        mlm_y_loss['train'] = []
        mlm_y_loss['val'] = []

        # len(loader) # Number of batches
        # len(loader.dataset) # Number of samples in dataset

        for epoch in range(mlm_params["epochs"]):
            # setup loop with TQDM and dataloader
            running_loss = 0.0
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

                running_loss += loss.item() * len(batch['input_ids'])

            epoch_loss = running_loss / len(loader.dataset)
            mlm_y_loss["train"].append(epoch_loss)

        if not transferLearning:
            endTime = datetime.now()
            elapsedTime = endTime - startTime

        if platform.system() == "Windows":
            model.save_pretrained(fr"D:\zProjects\MLM\Saved_Models\UMD_MLM_pretrain_{model_name}")
        elif platform.system() == "Linux":
            model.save_pretrained(fr"UMD_MLM_pretrain_{model_name}")

    ##-----Regular modeling-----
    if mainTask:

        # Models not for sequence classification are run in pytorch
        if modelType != "transformer":
            model.eval()

        if tf.test.gpu_device_name():
            print('GPU: {}'.format(tf.test.gpu_device_name()))
        else:
            print('CPU version')
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        startTime = datetime.now()

        # if boolDict["MultiTask"]: # If Multi-tasking

        # Sort multi task datasets
        # Only supports CSSRS, UMD, or Georgetown
        datasets["multitask"].sort()

        i = 0

        # Assumes that sorted list has CSSRS first
        main_df1 = importCSSRS(CSSRS_path, num_labels=CSSRS_n_label)
        # Make all text lowercase
        main_df1["Post"] = main_df1["Post"].apply(lambda x: x.lower())

        split_main_task_dataset_name = datasets["multitask"][1].split("-")
        if split_main_task_dataset_name[0] == "UMD":
            if split_main_task_dataset_name[1] == "crowd":
                task_train, task_test = importUMD(UMD_path, split_main_task_dataset_name[1], split_main_task_dataset_name[2])
            elif split_main_task_dataset_name[1] == "expert":
                task_train, task_test = importUMD(UMD_path, split_main_task_dataset_name[1])

            # For now, remove all posts with empty post_body (i.e. just post title)
            task_train = task_train[task_train["post_body"].notnull()]  # Removes 72 posts (task-train-A)
            task_test = task_test[task_test["post_body"].notnull()]  # Removes 5 posts (task-train-A)
            task_train = task_train.rename({"post_body":"Post", "label":"Label"}, axis=1)
            task_test = task_test.rename({"post_body":"Post", "raw_label":"Label"}, axis=1)
            main_df2 = pd.concat([task_train, task_test], ignore_index=True)
            main_df2,_ = getLabels(main_df2, 4, "UMD")


        df1 = pd.DataFrame({"Post": main_df1["Post"], "Label": main_df1["Label"]}, columns=['Post', 'Label']) #CSSRS
        df2 = pd.DataFrame({"Post": main_df2["Post"], "Label": main_df2["Label"]}, columns=['Post', 'Label']) #UMD

        # number_to_sample = [84, 36, 90, 290]
        # df2 = df2.groupby("Label").apply(lambda x: x.sample(n=number_to_sample[x["Label"].unique().item()], random_state=seed))
        # df2 = df2.reset_index(drop=True)

        # Create alias for principal embedding
        # Holdover from previous code. Will remove once I remove all the principal conceptnet embedding code

        if boolDict["CV"] == True:
            # Define per-fold score containers
            acc_per_fold1 = []
            loss_per_fold1 = []
            fold_stats1 = []
            fold_matrix1 = []
            test_size_per_fold1 = []

            acc_per_fold2 = []
            loss_per_fold2 = []
            fold_stats2 = []
            fold_matrix2 = []
            test_size_per_fold2 = []

            fold_hyper = []

            #Initialize dataframe for overall results
            whole_results = pd.DataFrame({"Actual_CSSRS": pd.Series(dtype=int), "Predicted_CSSRS": pd.Series(dtype=int),
                                          "PredictedProba_CSSRS": pd.Series(dtype=int), "Fold_CSSRS": pd.Series(dtype=int),
                                          "Actual_UMD": pd.Series(dtype=int), "Predicted_UMD": pd.Series(dtype=int),
                                          "PredictedProba_UMD": pd.Series(dtype=int), "Fold_UMD": pd.Series(dtype=int)})
            fold_results = []

            # Initalize stratified K-Fold splitting function
            sfk = StratifiedKFold(n_splits=n_folds, shuffle=False)
            fold_num = 1

            # Shuffle UMD dataset
            df2 = df2.sample(frac=1, random_state=seed).reset_index(drop=True)

            # Perform actuall splitting
            for (train_indx1, test_indx1), (train_indx2, test_indx2) in zip(sfk.split(df1["Post"], df1["Label"]), sfk.split(df2["Post"], df2["Label"])):
                train_val_sets = {}

                # Obtain train fold for first dataset
                fold_train1 = df1.iloc[train_indx1].copy()
                fold_train1 = fold_train1.reset_index(drop=True)
                # Obtain train fold for second dataset
                fold_train2 = df2.iloc[train_indx2].copy()
                fold_train2 = fold_train2.reset_index(drop=True)

                # Implement sampling k from each class for few-shot learning
                if few_shot["bool"]:
                    fold_train_K = fold_train.groupby("Label").sample(n=few_shot["k"], random_state=seed)

                    fold_val = fold_train.drop(fold_train_K.index, axis=0)
                    fold_val = fold_val.reset_index(drop=True)
                    X_val_fold = fold_val["Post"]
                    y_val_fold = fold_val["Label"]
                    y_val_fold  = tf.convert_to_tensor(y_val_fold)

                    fold_train = fold_train_K
                    fold_train = fold_train.sample(frac=1).reset_index(drop=True)
                else:
                    X_val_fold1 = ""
                    X_val_fold2 = ""
                    y_val_fold1 = ""
                    y_val_fold2 = ""

                # Split train fold into text and labels for first training set
                X_train_fold1 = fold_train1["Post"]
                y_train_fold1 = fold_train1["Label"]
                # Split train fold into text and labels for second training set
                X_train_fold2 = fold_train2["Post"]
                y_train_fold2 = fold_train2["Label"]

                # Obtain test fold and split into input and labels for first dataset
                fold_test1 = df1.iloc[test_indx1].copy()
                fold_test1 = fold_test1.reset_index(drop=True)
                X_test_fold1 = fold_test1["Post"]
                y_test_fold1 = fold_test1["Label"]

                # Obtain test fold and split into input and labels for second dataset
                fold_test2 = df2.iloc[test_indx2].copy()
                fold_test2 = fold_test2.reset_index(drop=True)
                X_test_fold2 = fold_test2["Post"]
                y_test_fold2 = fold_test2["Label"]

                # print(y_train_fold.value_counts())
                # print(y_test_fold.value_counts())

                # Convert train and test labels to tensors
                # if model_name.upper() != "SBERT":
                #     y_train_fold = tf.convert_to_tensor(y_train_fold)
                #     y_test_fold = tf.convert_to_tensor(y_test_fold)

                y_train_fold1 = tf.convert_to_tensor(y_train_fold1)
                y_test_fold1 = tf.convert_to_tensor(y_test_fold1)

                y_train_fold2 = tf.convert_to_tensor(y_train_fold2)
                y_test_fold2 = tf.convert_to_tensor(y_test_fold2)


                # Generate a print
                print('------------------------------------------------------------------------')
                print(f'Training for fold {fold_num} ...')

                if platform.system() == "Windows":
                    model_path = fr"D:\zProjects\MLM\Saved_Models\UMD_MLM_pretrain_{model_name}"
                elif platform.system() == "Linux":
                    model_path = fr"UMD_MLM_pretrain_{model_name}"

                # Get Model and Tokenizer
                model = getMultiTaskModel(model_name=model_name, model_path=model_path, modelType=modelType,
                                         transferLearning=transferLearning, CSSRS_n_label=CSSRS_n_label)

                tokenizer = getTokenizer(model_name)

                # Run fold

                nnModel, history, scores, \
                y_pred_proba, hyperparameters = MultiTaskrunFold(outputPath=outputPath, filespath=CSSRS_path,
                                                        model=model, tokenizer=tokenizer,
                                                        model_name=model_name, modelType=modelType,
                                                        max_length=max_length, num_labels=CSSRS_n_label,
                                                        hyperparameters=hyperparameters, n_folds=n_folds,
                                                        mlm_params=mlm_params, mlm_pretrain_bool=mlm_pretrain,
                                                        fold_num=fold_num, datasets=datasets, mask_strat=mask_strat,
                                                        X_train_fold1=X_train_fold1, X_train_fold2=X_train_fold2,
                                                        y_train_fold1=y_train_fold1, y_train_fold2=y_train_fold2,
                                                        X_val_fold1=X_val_fold1, X_val_fold2=X_val_fold2,
                                                        y_val_fold1=y_val_fold1, y_val_fold2=y_val_fold2,
                                                        X_test_fold1=X_test_fold1, X_test_fold2=X_test_fold2,
                                                        y_test_fold1=y_test_fold1, y_test_fold2=y_test_fold2,
                                                        boolDict=boolDict, few_shot=few_shot)

                train_CSSRS_auc = history.history['CSSRS_Output_auc']
                train_UMD_auc = history.history['UMD_Output_auc']
                val_CSSRS_auc = history.history['val_CSSRS_Output_auc']
                val_UMD_auc = history.history['val_UMD_Output_auc']

                train_CSSRS_acc = history.history['CSSRS_Output_accuracy']
                train_UMD_acc = history.history['UMD_Output_accuracy']
                val_CSSRS_acc = history.history['val_CSSRS_Output_accuracy']
                val_UMD_acc = history.history['val_UMD_Output_accuracy']

                train_loss = history.history['loss']
                val_loss = history.history['val_loss']

                train_CSSRS_loss = history.history['CSSRS_Output_loss']
                train_UMD_loss = history.history['UMD_Output_loss']
                val_CSSRS_loss = history.history['val_CSSRS_Output_loss']
                val_UMD_loss = history.history['val_UMD_Output_loss']
                epochs = range(len(train_CSSRS_auc))

                fold_results.append(
                    {"train_CSSRS_auc": train_CSSRS_auc, "val_CSSRS_auc": val_CSSRS_auc, "train_CSSRS_acc": train_CSSRS_acc, "val_CSSRS_acc": val_CSSRS_acc,
                     "train_CSSRS_loss": train_CSSRS_loss, "val_CSSRS_loss": val_CSSRS_loss,
                     "train_UMD_auc": train_UMD_auc, "val_UMD_auc": val_UMD_auc, "train_UMD_acc": train_UMD_acc, "val_UMD_acc": val_UMD_acc,
                     "train_UMD_loss": train_UMD_loss, "val_UMD_loss": val_UMD_loss,"epochs": epochs})

                # Generate generalization metrics
                print(
                    f'CSSRS Score for fold {fold_num}: {nnModel.metrics_names[1]} of {scores[1]}; {nnModel.metrics_names[3]} of {scores[3] * 100}%')
                print(
                    f'UMD Score for fold {fold_num}: {nnModel.metrics_names[2]} of {scores[2]}; {nnModel.metrics_names[5]} of {scores[5] * 100}%')

                acc_per_fold1.append(scores[3] * 100)
                acc_per_fold2.append(scores[5] * 100)

                loss_per_fold1.append(scores[1])
                loss_per_fold2.append(scores[2])

                fold_hyper.append(hyperparameters)

                if model_name.upper() == "SBERT":
                    list_probs = list(map(softmax, y_pred_proba.logits))
                    list_probs = [l.tolist() for l in list_probs]
                else:
                    list_probs1 = y_pred_proba[0].tolist()
                    list_probs2 = y_pred_proba[1].tolist()

                y_pred1 = np.argmax(list_probs1, axis=1)
                y_pred2 = np.argmax(list_probs2, axis=1)

                whole_results = pd.concat(
                    [whole_results, pd.DataFrame({"Actual_CSSRS": y_test_fold1.numpy().tolist(),
                                                  "Predicted_CSSRS": y_pred1.tolist(),
                                                  "PredictedProba_CSSRS": list_probs1, "Fold_CSSRS": fold_num,
                                                  "Actual_UMD": y_test_fold2.numpy().tolist(), "Predicted_UMD": y_pred2.tolist(),
                                                  "PredictedProba_UMD": list_probs2, "Fold_UMD": fold_num})], ignore_index=True)

                print(classification_report(y_test_fold1, y_pred1))
                print(classification_report(y_test_fold2, y_pred2))


                # contains precision, recall, and f1 score for each class for CSSRS
                report_1 = classification_report(y_test_fold1, y_pred1, output_dict=True)
                # contains precision, recall, and f1 score for each class for UMD
                report_2 = classification_report(y_test_fold2, y_pred2, output_dict=True)


                # Get only precision, recall, f1-score, and support statistics
                # filtered_report = {str(label): report[str(label)] for label in range(num_labels)}

                matrix1 = confusion_matrix(y_test_fold1, y_pred1)
                print(f"{CSSRS_n_label}-label confusion matrix (CSSRS)")
                print(matrix1)

                matrix2 = confusion_matrix(y_test_fold2, y_pred2)
                print(f"4-label confusion matrix (UMD)")
                print(matrix2)

                # Increase Fold Number
                fold_num = fold_num + 1

                tf.keras.backend.clear_session()
                tf.random.set_seed(seed)
            # == Provide average scores ==
            print('------------------------------------------------------------------------')
            print('Score per fold')
            for i in range(0, len(acc_per_fold1)):
                print('------------------------------------------------------------------------')
                print(f'> Fold {i + 1} - CSSRS - Loss: {loss_per_fold1[i]} - Accuracy: {acc_per_fold1[i]}%')
                print(f'> Fold {i + 1} - UMD - Loss: {loss_per_fold2[i]} - Accuracy: {acc_per_fold2[i]}%')
            print('------------------------------------------------------------------------')
            print('Average scores for all folds:')
            print(f'> CSSRS - Accuracy: {np.mean(acc_per_fold1)} (+- {np.std(acc_per_fold1)})')
            print(f'> CSSRS - Loss: {np.mean(loss_per_fold1)}')
            print(f'> UMD - Accuracy: {np.mean(acc_per_fold2)} (+- {np.std(acc_per_fold2)})')
            print(f'> UMD - Loss: {np.mean(loss_per_fold2)}')
            print('------------------------------------------------------------------------')

            overallResults1 = getStatistics(outputPath, whole_results["Actual_CSSRS"], whole_results["PredictedProba_CSSRS"],
                                           whole_results["Predicted_CSSRS"], CSSRS_n_label)
            overallResults2 = getStatistics(outputPath, whole_results["Actual_UMD"], whole_results["PredictedProba_UMD"],
                                           whole_results["Predicted_UMD"], 4)


            # whole_results.to_csv(os.path.join(outputPath, "Actual_vs_Predicted.csv"), index=False)
            endTime = datetime.now()
            elapsedTime = endTime - startTime
            outputFileName = f"MultiTask OverallResults {CSSRS_n_label}Label.csv"

        else:
            # pass
            # y_train = tf.convert_to_tensor(y_train)
            # y_test = tf.convert_to_tensor(y_test)

            nnModel, history, scores, y_pred_proba, hyperparameters = MultiTaskrunFold(outputPath=outputPath, filespath=CSSRS_path,
                                                                                  model=model, tokenizer=tokenizer,
                                                                                  model_name=model_name, modelType=modelType,
                                                                                  max_length=max_length, num_labels=CSSRS_n_label,
                                                                                  hyperparameters=hyperparameters,
                                                                                  n_folds=n_folds, fold_num=1,
                                                                                  mlm_pretrain_bool = mlm_pretrain,
                                                                                  datasets=datasets, mask_strat=mask_strat,
                                                                                  X_train_fold=X_train, y_train_fold=y_train,
                                                                                  X_test_fold=X_test, y_test_fold=y_test,
                                                                                  boolDict=boolDict, few_shot=few_shot)

            if model_name.upper() != "SBERT":
                list_probs = list(map(softmax, y_pred_proba.logits))
                list_probs = [l.tolist() for l in list_probs]
            else:
                list_probs = y_pred_proba
            y_pred = np.argmax(list_probs, axis=1)

            whole_results = pd.DataFrame({"Actual": y_test.numpy().tolist(), "Predicted": y_pred.tolist(),
                                          "PredictedProba": y_pred_proba.tolist()})

            overallResults = getStatistics(outputPath, whole_results["Actual"], whole_results["PredictedProba"],
                                           whole_results["Predicted"], CSSRS_n_label)
            fold_results = []
            # printPredictions(y_test, y_pred, num_labels, outputPath)
            endTime = datetime.now()
            elapsedTime = endTime - startTime

            outputFileName = f"OverallResults {CSSRS_n_label}Label (no CV).csv"

    if not mlm_pretrain:
        mlm_y_loss = {}

    # If only doing MLM pre-training and not running main task
    if mlm_pretrain == True and mainTask == False:
        new_output_path = os.path.join(outputPath, "[MLM Loss]")
        if not os.path.exists(new_output_path):
            os.mkdir(new_output_path)
        printMLMResults(outputPath=new_output_path, model_name=model_name, mask_strat=mask_strat, mlm_y_loss=mlm_y_loss,
                        mlm_params=mlm_params)
    else:
        printOverallResultsMultiTask(outputPath=outputPath, fileName=outputFileName, mainTaskBool=mainTask,
                            n_label=CSSRS_n_label, max_length=max_length, boolDict=boolDict,
                            numCV=n_folds, model_type=modelType, stats1=overallResults1, stats2=overallResults2,
                            pretrain_bool=mlm_pretrain, hyperparameters=fold_hyper, execTime=elapsedTime, whole_results=whole_results,
                            fold_results=fold_results, model_name=model_name, datasets=datasets, mask_strat=mask_strat,
                            mlm_params=mlm_params, transferLearning=transferLearning, mlm_y_loss=mlm_y_loss, few_shot=few_shot)