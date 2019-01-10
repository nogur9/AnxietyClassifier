from create_classifier.save_best_model import load_classifier_pipeline, load_prepossessing_pipeline
import pandas as pd
import os

def load_clf(date):

    prepro = load_prepossessing_pipeline(f"C:\‏‏PycharmProjects\AnxietyClassifier\create_classifier\models\prepossessing pipeline_final_100K{date}.joblib")
    clf = load_classifier_pipeline(f"C:\‏‏PycharmProjects\AnxietyClassifier\create_classifier\models\classifier pipeline_final_100K{date}.joblib")
    return prepro, clf


def load_data():
    # features file of the subjeect's data

    dir_path = "C:\‏‏PycharmProjects\AnxietyClassifier\\test_data"
    #file_name = "extracted_eye_link_features_subjects__[998 999]_2018-12-02.xlsx"
    file_name = r"extracted_eye_link_features_subjects__2019-01-07.xlsx"
    df = pd.read_excel(os.path.join(dir_path, file_name), sheet_name='Sheet1')
    return df

def predict(x1):
    x1 = x1.reshape((1, -1))
    prepro, clf = load_clf("2018-12-03")
    x1 = prepro.transform(x1)
    y_pred = clf.predict(x1)
    return y_pred


def split_to_subjects(df):
    sub_num_col = 'Subject_Number'
    sub_num = df[sub_num_col]
    X = df.drop([sub_num_col], 1)
    Y_predicted = []
    for i in range(X.shape[0]):
        xi = X.iloc[i]
        Y_predicted.append(predict(xi))
    df2 = pd.DataFrame({"subject": sub_num.values, "predicted_class": Y_predicted})
    return df2


def from_classes_to_labels(class_num):
    if class_num:
        return 'high'
    else:
        return 'low'
    #dataset[[group_column_name]] = dataset[[group_column_name]].replace('low', 0)
    #dataset[[group_column_name]] = dataset[[group_column_name]].replace('high', 1)
    #dataset[[group_column_name]] = dataset[[group_column_name]].replace('clinical', 1)


def main():
    df = load_data()
    results = split_to_subjects(df)
    results['group'] = results.predicted_class.apply(from_classes_to_labels)
    print(results)
    results.to_excel("100K_results_extracted_eye_link_features_subjects__2019-01-07.xlsx")




main()