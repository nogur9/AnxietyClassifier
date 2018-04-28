


def test_set(dataset):
    """
    :param dataset: Pandas DataFrame dataset  

    Uses - linear SVM, median imputation and RFE feature selection. 
    sclaes the training and the test data by minMax scaler and prints the followings:
    1. features importance. where 1 means XXX and the rest means XXX.
    2. Accuracy on test set
    3. Confusion Matrix
    4. Precision and recall
    """

    tmp_dataset = imputing_median(dataset)
    X_train, X_test, Y_train, Y_test = split_data(tmp_dataset)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    kfold = model_selection.KFold(n_splits=3)
    linear_svm = SVC(kernel='linear')
    rfecv_SVM = RFECV(estimator=linear_svm, step=1, cv=kfold, scoring=SCORING)
    rfecv_SVM.fit(X_train_scaled, Y_train)

    names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
             "Age"]
    svm_feature_importance = sorted([(rfecv_SVM.ranking_[i], names[i]) for i in range(len(names))])
    print ("Svm_feature_importance {}".format(svm_feature_importance))

    # Make predictions on test set
    SVM_predictions = rfecv_SVM.predict(X_test_scaled)
    print("Accuracy on test set - {}".format(accuracy_score(Y_test, SVM_predictions)))
    print("Confusion Matrix (predicted class X actual class)\n {}".format(confusion_matrix(Y_test, SVM_predictions)))
    print("Classification Report\n {}".format(classification_report(Y_test, SVM_predictions)))
