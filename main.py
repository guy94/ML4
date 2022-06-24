import time
import sklearn_relief as relief
import mrmr
import scipy.io
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SelectFdr, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeavePOut
from BorutaShap import BorutaShap
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score

random_forest = RandomForestClassifier(min_samples_leaf=2, max_depth=13)
svm = SVC()
knn = KNeighborsClassifier(n_neighbors=5)
nb_classifier = GaussianNB()
logistic = LogisticRegression(random_state=0)

# models = [random_forest, svm, knn, nb_classifier, logistic]
MODELS = [nb_classifier]
# posible_k = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]
POSSIBLE_K = [3, 4]


def read_dbs():
    """

    :return:
    """
    allaml_mat = scipy.io.loadmat('data/db1/ALLAML.mat')
    allaml_df = pd.DataFrame(allaml_mat['X'])
    allaml_df['y'] = allaml_mat['Y']

    arcene_mat = scipy.io.loadmat('data/db1/arcene.mat')
    arcene_df = pd.DataFrame(arcene_mat['X'])
    arcene_df['y'] = arcene_mat['Y']
    #
    # Leukemia_4c_arff = arff.loadarff('data/db2/Leukemia_4c.arff')
    # Leukemia_4c_df = pd.DataFrame(Leukemia_4c_arff[0])
    #
    # Leukemia_3c_arff = arff.loadarff('data/db2/Leukemia_3c.arff')
    # Leukemia_3c_df = pd.DataFrame(Leukemia_3c_arff[0])

    # return {'ALLAML': allaml_df, 'arcene': arcene_df, 'Leukemia_3c_df': Leukemia_3c_df, 'Leukemia_4c_df': Leukemia_4c_df}
    return {'ALLAML': allaml_df, 'arcene': arcene_df}


def fill_na(df):
    """
    fills NaN values for each of the split data sets.
    Categorical columns are filled with the most common value,
    while numerical columns are filed with the mean of the column.
    :return:
    """

    for col in df:
        if df[col].dtype == np.object:
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)


def discretization(df):
    """
     Not all sklearn models can handle categorical data, label-encoding (similar to dummy data) was applied.
     in addition, we used discretization for continuous-numerical data.
    :return:
    """

    label_encoder = LabelEncoder()
    amount_of_bins = 3

    for col in df:
        if df[col].dtype == 'categorical':
            df[col] = label_encoder.fit_transform(df[col])

        if df[col].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
            est = KBinsDiscretizer(n_bins=amount_of_bins, encode='ordinal', strategy='uniform')
            values = df[col].values.reshape(-1, 1)
            est.fit(values)
            df[col] = est.transform(values)


def iterate_dbs(dbs, fs_methods):
    """

    :param dbs:
    :return:
    """

    for df_name, df in dbs.items():
        df.rename(columns=lambda x: str(x), inplace=True)
        cv_method, n_splits_cv, is_select_k_best = choose_method_for_cross_validation(df)
        kf = cv_method(n_splits=n_splits_cv)
        kf = KFold(n_splits=2)
        df.columns = [*df.columns[:-1], 'y']
        X = df.loc[:, df.columns != 'y']
        y = df['y']

        if is_select_k_best:
            selector = SelectKBest(f_classif, k=1000).fit(X, y)
            cols = selector.get_support(indices=True)
            X = X.iloc[:, cols]

        accumulated_preds = {}  # {[model]: {k: preds}}
        accumulated_y_test = {}  # {k: y_test}
        run_times = {'fs_method': {}, 'fit': {}, 'predict': {}}

        ###################
        # fill_na(X)
        # y.fillna(999, inplace=True)
        # discretization(X)
        ###################

        n_iters = 0

        for fs_method in fs_methods:
            k_and_features_to_keep_dict = {}
            last_k = -1
            fs_method_name = fs_method.__name__
            run_times['fs_method'][fs_method_name] = {}

            selector = ''
            for k in POSSIBLE_K:
                start_time_fs_method = time.time()

                if fs_method_name == 'SelectFdr':
                    selector = SelectFdr(alpha=0.1).fit(X=X, y=y)
                    features_and_scores = get_features_scores(selector.scores_, X, k)
                    cols = list(features_and_scores.keys())

                elif fs_method_name == 'sklearn_relief':
                    selector = relief.Relief(n_features=k)
                    selector = selector.fit(X.values, X.columns)
                    features_and_scores = get_features_scores(selector.w_, X, k)
                    cols = list(features_and_scores.keys())

                elif fs_method_name == 'mrmr':
                    selector = mrmr.mrmr_classif(X=X, y=y, K=k, return_scores=True)
                    features_and_scores = get_features_scores(selector[1], X, k)
                    cols = list(features_and_scores.keys())

                elif fs_method_name == 'f_classif':
                    selector = SelectKBest(f_classif, k=k).fit(X, y)
                    features_and_scores = get_features_scores(selector.scores_, X, k)
                    cols = list(features_and_scores.keys())

                elif fs_method_name == 'run_shap':
                    pass
                    features_and_scores = get_features_scores(selector.scores_, cols, X, k)
                    cols = list(features_and_scores.keys())

                end_time_fs_method = time.time()
                fs_method_run_time = (end_time_fs_method - start_time_fs_method)
                run_times['fs_method'][fs_method_name][k] = fs_method_run_time

                k_and_features_to_keep_dict[k] = features_and_scores
                new_X = X[cols]

                for train_index, test_index in kf.split(new_X, y):
                    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], \
                                                       y.iloc[train_index], y.iloc[test_index]

                    fill_na(X_train)
                    fill_na(X_test)
                    y_train.fillna(999, inplace=True)
                    y_test.fillna(999, inplace=True)

                    discretization(X_train)
                    discretization(X_test)

                    if not last_k == k:
                        run_models(X_train, y_train, X_test, y_test, k, accumulated_preds, accumulated_y_test, run_times)
                        last_k = k
                    n_iters += 1

                    # for k, features_dict in k_and_features_to_keep_dict.items():
                    #     features = list(features_dict.keys())
                    #     if not last_k == k:
                    #         new_X_train = X_train[features]
                    #         new_X_test = X_test[features]
                    #         run_models(new_X_train, y_train, new_X_test, y_test, k, accumulated_preds, accumulated_y_test, run_times)
                    #     last_k = k
                    # n_iters += 1

                evaluations = evaluate_models(accumulated_preds, accumulated_y_test)
                export_data(df_name, k_and_features_to_keep_dict, run_times, evaluations, cv_method, n_splits_cv,
                            df, fs_method.__name__)


def get_features_scores(scores, df, k):
    cols = []
    scores = scores.tolist()
    temp = sorted(scores)[-k:]
    for elem in temp:
        cols.append(scores.index(elem))
    features_and_scores = {}

    for col in cols:
        score = scores[col]
        col_name = df.columns[col]
        features_and_scores[col_name] = score
    return features_and_scores


def choose_method_for_cross_validation(df):
    """

    :param df:
    :return:
    """

    df_count = len(df)

    if df_count < 50:
        return LeavePOut, 2, True
    elif 50 <= df_count < 100:
        return KFold, df_count, True
    elif 100 <= df_count < 1000:
        return KFold, 5, False
    elif 1000 <= df_count:
        return KFold, 10, False


def run_shap(X, y):
    """

    :param X:
    :param y:
    :return:
    """

    selector = BorutaShap(importance_measure='shap', classification=True)
    selector.fit_transform(X=X, y=y, n_trials=20, sample=False, verbose=True, normalize=True)
    accepted_features_scores = selector.X_feature_import
    accepted_features_names = selector.X.columns.values
    names_and_scores_dict = dict(zip(accepted_features_names, accepted_features_scores))

    k_and_features_dict = {}
    sorted_features = sorted(names_and_scores_dict, key=names_and_scores_dict.get, reverse=True)

    # for k in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]:
    for k in POSSIBLE_K:
        selected_features = sorted_features[:k]
        features_and_scores = {}
        for feature in selected_features:
            features_and_scores[feature] = names_and_scores_dict[feature]

        k_and_features_dict[k] = features_and_scores

    return k_and_features_dict

    # selector = BorutaShap(importance_measure='shap', classification=True)
    # selector.fit(X=X, y=y, n_trials=20, sample=False, verbose=True, normalize=True)
    # accepted_features_scores = selector.X_feature_import
    # accepted_features_names = selector.X.columns.values
    # names_and_scores_dict = dict(zip(accepted_features_names, accepted_features_scores))
    #
    # k_and_features_dict = {}
    # sorted_features = sorted(names_and_scores_dict, key=names_and_scores_dict.get, reverse=True)
    #
    # # for k in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]:
    # for k in POSSIBLE_K:
    #     selected_features = sorted_features[:k]
    #     features_and_scores = {}
    #     for feature in selected_features:
    #         features_and_scores[feature] = names_and_scores_dict[feature]
    #
    #     k_and_features_dict[k] = features_and_scores
    #
    # return k_and_features_dict


def run_models(X_train, y_train, X_test, y_test, k, accumulated_preds, accumulated_y_test, run_times):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """

    if not k in accumulated_y_test:
        accumulated_y_test[k] = []
    accumulated_y_test[k] += y_test.tolist()

    for model in MODELS:
        model_name = type(model).__name__
        start_time_fit_method = time.time()
        model.fit(X_train, y_train)
        end_time_fit_method = time.time()
        fit_method_run_time = (end_time_fit_method - start_time_fit_method)
        if model_name not in run_times['fit']:
            run_times['fit'][model_name] = {}
        run_times['fit'][model_name][k] = fit_method_run_time

        start_time_predict_method = time.time()
        y_pred = model.predict(X_test)
        end_time_predict_method = time.time()
        predict_method_run_time = (end_time_predict_method - start_time_predict_method)
        if model_name not in run_times['predict']:
            run_times['predict'][model_name] = {}
        run_times['predict'][model_name][k] = predict_method_run_time

        if not model_name in accumulated_preds:
            accumulated_preds[model_name] = {}

        if not k in accumulated_preds[model_name]:
            accumulated_preds[model_name][k] = []

        accumulated_preds[model_name][k] += y_pred.tolist()


def evaluate_models(accumulated_preds, accumulated_y_test):
    """

    :param accumulated_preds:
    :param accumulated_y_test:
    :param num_of_iterations:
    :return:
    """

    models_scores = {}
    for model_name, k_and_preds in accumulated_preds.items():
        for k, y_preds in k_and_preds.items():

            acc = accuracy_score(accumulated_y_test[k], y_preds)
            mcc = matthews_corrcoef(accumulated_y_test[k], y_preds)
            auc_roc = roc_auc_score(accumulated_y_test[k], y_preds)
            pr_auc = average_precision_score(accumulated_y_test[k], y_preds)

            if not model_name in models_scores:
                models_scores[model_name] = {}

            if not k in models_scores[model_name]:
                models_scores[model_name][k] = {}

            models_scores[model_name][k]['acc'] = round(acc, 3)
            models_scores[model_name][k]['mcc'] = round(mcc, 3)
            models_scores[model_name][k]['auc_roc'] = round(auc_roc, 3)
            models_scores[model_name][k]['pr_auc'] = round(pr_auc, 3)

    return models_scores


def export_data(df_name, k_and_features_to_keep_dict, run_times, evaluations, cv_method, n_splits_cv, df, fs_method):
    db_rows_data = []

    for model in MODELS:
        model_name = type(model).__name__
        for k, features_dict in k_and_features_to_keep_dict.items():
            features = list(features_dict.keys())
            features_scores = list(features_dict.values())

            fs_method_time = run_times['fs_method'][fs_method][k]
            fit_method_time = run_times['fit'][model_name][k]
            predict_method_time = run_times['predict'][model_name][k]

            for score_method, score in evaluations[model_name][k].items():
                single_row_data = [df_name, len(df.index), len(df.columns), fs_method, model_name,
                                   k, cv_method.__name__, n_splits_cv, score_method, score, str(features),
                                   features_scores,
                                   fs_method_time, fit_method_time, predict_method_time]
                print(single_row_data)
                db_rows_data.append(single_row_data)

    df = pd.DataFrame(data=db_rows_data)
    df.to_csv('output.csv', mode='a', header=False, index=False)


if __name__ == '__main__':
    # 'mRMR', 'f_classif', 'SelectFdr', 'ReliefF'
    final_df = pd.DataFrame(columns=['Dataset name', 'Number of samples', 'Original number of features',
                                     'Filtering algorithm', 'Learning algorithm', 'Number of features selected', 'CV method', 'Fold',
                                     'Measure type', 'Measure value', 'List of selected features names (long STRING)',
                                     'Selected features scores', 'Feature selection run time', 'Fit run time',
                                     'Predict run time'])
    final_df.to_csv('output.csv', index=False)

    fs_methods = [mrmr, SelectFdr, relief, f_classif]
    dbs = read_dbs()
    iterate_dbs(dbs, fs_methods)
