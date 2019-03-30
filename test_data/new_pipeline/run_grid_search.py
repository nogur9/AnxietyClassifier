
from create_classifier.build_grid_search_pipeline import build_pipeline, get_params_grid,build_full_pipeline, get_full_params_grid
from create_classifier.get_data import get_data, get_data_for_reg
from create_classifier.get_grid_search_results import meow, linear_regressor, get_reasults_only_clf_pipeline
import datetime




file_name = r"training_set.xlsx"



print("\n{}, RF FI #iter = 100000_saving model\n\n".format(datetime.datetime.now().strftime('%Y-%m-%d')))



X, Y = get_data(file_name, LSAS_threshold = 50)
prepro,clf = build_full_pipeline()
prepro_params, params_grid = get_full_params_grid()
meow(X, Y, prepro, clf,prepro_params, params_grid)

reg = 0
if reg:
    X, Y = get_data_for_reg(file_name)
    prepro, clf = build_full_pipeline()
    prepro_params, params_grid = get_full_params_grid()
    linear_regressor(X, Y, prepro)
