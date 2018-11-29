
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.linear_model import ElasticNet, LinearRegression
from dim_reduction.feature_selection_pipeline import interactive_pipeline
import itertools
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, KFold
from create_classifier.save_best_model import save_classifier_pipeline, save_prepossessing_pipeline


def meow (X, Y,prepro, pipeline,prepro_params, params_grid, saving = 1):

    scores = ['accuracy']

    for score in scores:
        #KFold(n_splits=10),, LeaveOneOut()
        for cv in [LeaveOneOut()]:
            success_rate_list = []
            print("# Tuning hyper-parameters for %s" % cv)
            print("prepossessing params")
            hyper_params = [list(X.items())[0][1] for X in prepro_params]
            hyper_params_names = [list(x)[0].split("__") for x in prepro_params]
            for iter in list(itertools.product(*hyper_params)):

                prepro_hyper_params = iter
                for value, hyperparam in zip(iter, hyper_params_names):
                    prepro.named_steps[hyperparam[0]].__dict__[hyperparam[1]] = value
                copy_X = prepro.fit_transform(X, Y)

                clf = GridSearchCV(pipeline, params_grid, cv=cv, scoring=score)
                clf.fit(copy_X, Y)
                success_rate_list.append((clf.best_score_, clf.cv_results_['std_test_score'][clf.best_index_], clf.best_estimator_, prepro_hyper_params))

            best_success_rate = sorted(success_rate_list, key=lambda x:x[0], reverse=True)[0]
            print(best_success_rate[0],"clf.best_score_",best_success_rate[1], "clf.cv_results_['std_test_score'][clf.best_index_]",
                  best_success_rate[2], "clf.best_estimator_", "prepro_hyper_params",best_success_rate[3], sep="\n")

            if saving:
                for value, hyperparam in zip(best_success_rate[3], hyper_params_names):
                    prepro.named_steps[hyperparam[0]].__dict__[hyperparam[1]] = value
                save_classifier_pipeline(prepro)
                save_prepossessing_pipeline(best_success_rate[2])


def get_reasults(X, Y,prepro, pipeline, param_grid, save=None):
    scores = ['accuracy']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        prepro.named_steps['pca'].n_components = 3
        print(prepro.named_steps['pca'].n_components)
        X = prepro.fit_transform(X, Y)

        loo =LeaveOneOut()
        print(X.shape)
        clf = GridSearchCV(pipeline, param_grid, cv=loo, scoring=score, verbose=True)
        clf.fit(X, Y)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("mean - %0.3f (std - %0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))

        print()
        print("best score - ",clf.best_score_ )
        print("best STD", clf.cv_results_['std_test_score'][clf.best_index_])


def get_reasults_only_clf_pipeline(X, Y, clf_pipeline, clf_param_grid, save=None):
    for pca_n in [5,7]:
        for rf_k in [5,7]:
            X_train, Y_train = interactive_pipeline(X, Y, pca_n+2,pca_n+rf_k+2)
            scores = ['accuracy']
            for score in scores:
                print("# Tuning hyper-parameters for %s" % score)
                print("pca n", pca_n + 2, "RF k", pca_n + rf_k + 2)
                print()
                loo = LeaveOneOut()
                clf = GridSearchCV(clf_pipeline, clf_param_grid, cv=loo, scoring=score)
                clf.fit(X, Y)
                print("Best parameters set found on development set:")
                print()
                print(clf.best_estimator_)
                print()
                print("Grid scores on development set:")
                print()
                for params, mean_score, scores in clf.grid_scores_:
                    print("mean - %0.3f (std - %0.03f) for %r"
                          % (mean_score, scores.std() , params))
                print()
                print("best score - ", clf.best_score_)
                print("best STD", clf.cv_results_['std_test_score'][clf.best_index_])
                print("naive classifier score {}".format(max(Y_train[Y_train == 0].count(), Y_train[Y_train == 1].count())/len(Y)))


def linear_regressor(X, Y, prepro):
    #prepro
    X = prepro.fit_transform(X, Y)

    lin_reg = LinearRegression()
    lin_reg = lin_reg.fit(X, Y)
    Y_pred = lin_reg.predict(X)
    print("Linear reggression score")
    print(lin_reg.score(X, Y))
    print(np.sqrt(mean_squared_error(Y, Y_pred)))

    print("ElasticNet")
    enet = Pipeline([("reg", ElasticNet())])
    enet_params = [{"reg__alpha": [0,0.5,1]}, {"reg__l1_ratio":[0, 0.5, 1]}]
    reg = GridSearchCV(enet, enet_params, scoring='neg_mean_squared_error')
    reg.fit(X, Y)
    print("Best parameters set found on development set:")
    print(reg.best_estimator_)
    print("best score - ", np.sqrt(-reg.best_score_))
    print("best STD", reg.cv_results_['std_test_score'][reg.best_index_])
