from joblib import dump, load
import datetime


def save_prepossessing_pipeline(pipe):
    file_name = "models/prepossessing pipeline_{}".format(datetime.datetime.now().strftime('%Y-%m-%d'))
    dump(pipe, file_name+'.joblib')


def save_classifier_pipeline(clf):
    file_name = "models/classifier pipeline_{}".format(datetime.datetime.now().strftime('%Y-%m-%d'))
    dump(clf, file_name+'.joblib')


def load_prepossessing_pipeline(file_name):
    pipe = load(file_name)
    return pipe


def load_classifier_pipeline(file_name):
    clf = load(file_name)
    return clf

