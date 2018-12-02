from create_classifier.save_best_model import load_classifier_pipeline, load_prepossessing_pipeline
import pandas as pd
import os

def load_clf(date):

    prepro = load_prepossessing_pipeline(f"C:\‏‏PycharmProjects\AnxietyClassifier\create_classifier\models\prepossessing pipeline_{date}.joblib")
    clf = load_classifier_pipeline(f"C:\‏‏PycharmProjects\AnxietyClassifier\create_classifier\models\classifier pipeline_{date}.joblib")
    return prepro, clf


def load_features():
    pass
path = "C:\‏‏PycharmProjects\AnxietyClassifier\ExtractedFeaturesFiles"
file_name = "extracted_eye_link_features_subjects__[998 999]_2018-12-02.xlsx"
prepro, clf = load_clf("2018-12-02")
df = pd.read_excel(os.path.join(path, file_name), sheet_name='Sheet1')
sub_num_col = 'Subject_Number'
X = df.drop([sub_num_col], 1)
x1 = X.iloc[0]
x1 = x1.reshape((1, -1))
x1 = prepro.transform(x1)
y_pred = clf.predict(x1)

print(y_pred)
