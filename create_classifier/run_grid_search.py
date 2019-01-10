
from create_classifier.build_grid_search_pipeline import build_pipeline, get_params_grid,build_full_pipeline, get_full_params_grid
from create_classifier.get_data import get_data, get_data_for_reg
from create_classifier.get_grid_search_results import meow, linear_regressor, get_reasults_only_clf_pipeline
outliers = 0
if outliers:
    file_name = r"extracted_features_subjects_set_Updated,with_outlier_subjects_False_with_9029,9014,2018-10-29.xlsx"
    print("with the outliers")
if not outliers:
    file_name = r"extracted_features_subjects_set_Updated,with_outlier_subjects_False,2018-11-28.xlsx"
    print("without the outliers")
    X, Y = get_data(file_name)
    #print("naive classifier score {}".format(max(Y[Y == 0].count(), Y[Y == 1].count()) / len(Y)))
    prepro, clf = build_full_pipeline()


print("\n03_12_2018, iter = 100000_finalrun\n\n")

clf = 0
if clf:
    prepro_params, params_grid = get_full_params_grid()
    meow(X, Y, prepro, clf,prepro_params, params_grid)

reg = 0
if reg:
    X, Y = get_data_for_reg(file_name)
    prepro, clf = build_full_pipeline()
    prepro_params, params_grid = get_full_params_grid()
    linear_regressor(X, Y, prepro)
cutoff = 1
if cutoff:
    for thresh in [50]:
        print("LSAS threshold - ", thresh)
        X, Y = get_data(file_name, LSAS_threshold = thresh)
        prepro,clf = build_full_pipeline()
        prepro_params, params_grid = get_full_params_grid()
        meow(X, Y, prepro, clf,prepro_params, params_grid)

iterative = 0
if iterative:
    X, Y = get_data(file_name)
    pipeline = build_pipeline()
    params_grid = get_params_grid()
    get_reasults_only_clf_pipeline(X, Y, pipeline, params_grid)
