import time
import sklearn_relief as relief
import mrmr
import scipy.io
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SelectFdr, chi2, RFE, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeavePOut
from BorutaShap import BorutaShap
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, PowerTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

random_forest = RandomForestClassifier(min_samples_leaf=2, max_depth=13)
svm = SVC()
knn = KNeighborsClassifier(n_neighbors=5)
nb_classifier = GaussianNB()
logistic = LogisticRegression(random_state=0)

# MODELS = [random_forest, svm, knn, nb_classifier, logistic]
MODELS = [nb_classifier]
# posible_k = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]
POSSIBLE_K = [2, 20]


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


def remove_variance_zero(df):
    """

    :param df:
    :return:
    """

    selector = VarianceThreshold(0)
    selector.fit(df)
    col_idx = selector.get_support(indices=True).tolist()
    cols = df.columns[col_idx]
    return df[cols]


def iterate_dbs(dbs, fs_methods):
    """

    :param dbs:
    :return:
    """

    for df_name, df in dbs.items():
        df.rename(columns=lambda x: str(x), inplace=True)
        cv_method, n_splits_cv, is_select_k_best = choose_method_for_cross_validation(df)
        # kf = cv_method(n_splits_cv)
        kf = KFold(2)
        df.columns = [*df.columns[:-1], 'y']
        X = df.loc[:, df.columns != 'y']
        y = df['y']
        power_transformer = PowerTransformer()

        if is_select_k_best and X.shape[1] > 1000:
            selector = SelectKBest(f_classif, k=1000).fit(X, y)
            cols = selector.get_support(indices=True)
            X = X.iloc[:, cols]

        run_times = {'fs_method': {}, 'fit': {}, 'predict': {}}

        n_iters = 0

        for fs_method in fs_methods:
            accumulated_preds = {}  # {[model]: {k: preds}}
            accumulated_y_test = {}  # {k: y_test}

            k_and_features_to_keep_dict = {}
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

                elif fs_method_name == 'RFE':
                    estimator = SVC(kernel="linear")
                    selector = RFE(estimator, n_features_to_select=10, step=1)
                    selector = selector.fit(X, y)
                    cols = selector.get_feature_names_out().tolist()
                    ranks = [1 for i in range(10)]
                    features_and_scores = dict(zip(cols, ranks))

                elif fs_method_name == 'run_shap':
                    all_features_and_scores = run_shap(X, y)
                    sorted_features = sorted(all_features_and_scores, key=all_features_and_scores.get, reverse=True)
                    selected_features = sorted_features[:k]
                    features_and_scores = {}
                    for feature in selected_features:
                        features_and_scores[feature] = all_features_and_scores[feature]

                    cols = list(features_and_scores.keys())

                end_time_fs_method = time.time()
                fs_method_run_time = (end_time_fs_method - start_time_fs_method)
                run_times['fs_method'][fs_method_name][k] = fs_method_run_time

                k_and_features_to_keep_dict[k] = features_and_scores
                new_X = X[cols]
                new_X = remove_variance_zero(new_X)

                for train_index, test_index in kf.split(new_X, y):
                    X_train, X_test, y_train, y_test = new_X.iloc[train_index], new_X.iloc[test_index], \
                                                       y.iloc[train_index], y.iloc[test_index]

                    fill_na(X_train)
                    fill_na(X_test)
                    y_train.fillna(999, inplace=True)
                    y_test.fillna(999, inplace=True)

                    discretization(X_train)
                    discretization(X_test)

                    # X_train = power_transformer.fit_transform(X_train)
                    # X_test = power_transformer.fit_transform(X_test)

                    run_models(X_train, y_train, X_test, y_test, k, accumulated_preds, accumulated_y_test, run_times)
                    n_iters += 1

            evaluations = evaluate_models(accumulated_preds, accumulated_y_test)
            export_data(df_name, k_and_features_to_keep_dict, run_times, evaluations, cv_method.__name__, n_splits_cv,
                        df, fs_method.__name__)


def get_features_scores(scores, df, k):
    cols = []
    scores = scores.tolist()
    d = sorted(scores, reverse=True)
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
        return KFold, df_count, True
    elif 50 <= df_count < 100:
        return LeavePOut, 2, True
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
    selector.fit(X=X, y=y, n_trials=20, sample=False, verbose=True, normalize=True)
    accepted_dict = dict(zip(selector.accepted, [1 for i in range(len(selector.accepted))]))
    tentative_dict = dict(zip(selector.tentative, [0.5 for i in range(len(selector.tentative))]))
    rejected_dict = dict(zip(selector.rejected, [0 for i in range(len(selector.rejected))]))

    return_dict = {**accepted_dict, **tentative_dict}
    return_dict = {**rejected_dict, **return_dict}

    return return_dict


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


def export_data(df_name, k_and_features_to_keep_dict, run_times, evaluations, cv_method_name, n_splits_cv, df, fs_method):
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
                                   k, cv_method_name, n_splits_cv, score_method, score, str(features),
                                   features_scores,
                                   fs_method_time, fit_method_time, predict_method_time]
                print(single_row_data)
                db_rows_data.append(single_row_data)

    df = pd.DataFrame(data=db_rows_data)
    df.to_csv('output.csv', mode='a', header=False, index=False)


def run_toy_example():

    data = pd.read_csv('data/toy/SPECTF.train', sep=",", header=None)
    data.columns = [i for i in range(data.shape[1])]
    data = data[data.columns[::-1]]
    # last = data[data.columns[-1]]
    # first = data.iloc[:, 0].copy()
    # data.iloc[:, 0] = last.copy()
    # data[data.columns[-1]] = first
    # data = data.transpose()
    return {'toy_example': data}


def friedman_test():
    """

    :return:
    """

    all_aucs = []
    algorithms_and_scores = {}
    df = pd.read_csv('output_backup.csv')
    grouped = df.groupby(['Dataset name', 'Filtering algorithm'])
    for name, group in grouped:
        aucs = group.loc[group['Measure type'] == 'auc_roc', 'Measure value']
        all_aucs.append(aucs)
        if name[1] not in algorithms_and_scores:
            algorithms_and_scores[name[1]] = []
        algorithms_and_scores[name[1]] += aucs.tolist()

    fs_result = friedmanchisquare(*all_aucs)

    # if fs_result.pvalue < 0.05:
    if fs_result.pvalue < 0.2:
        run_post_hoc(algorithms_and_scores)


def run_post_hoc(algorithms_and_scores):
    """

    :param df:
    :return:
    """

    df = pd.DataFrame.from_dict(algorithms_and_scores)
    long_df = pd.melt(df, var_name='filtering algorithm', value_name='auc score')

    print(sp.posthoc_dunn(long_df, val_col='auc score', group_col='filtering algorithm', p_adjust='bonferroni'))

    ranks_df = df.rank(axis=1, method='max', ascending=False)
    mean = ranks_df.mean(axis=0)
    print('\n')
    print(mean)


if __name__ == '__main__':

    friedman_test()

    # # 'mRMR', 'rfe', 'SelectFdr', 'ReliefF'
    # final_df = pd.DataFrame(columns=['Dataset name', 'Number of samples', 'Original number of features',
    #                                  'Filtering algorithm', 'Learning algorithm', 'Number of features selected', 'CV method', 'Fold',
    #                                  'Measure type', 'Measure value', 'List of selected features names (long STRING)',
    #                                  'Selected features scores', 'Feature selection run time', 'Fit run time',
    #                                  'Predict run time'])
    # final_df.to_csv('output.csv', index=False)
    # fs_methods = [RFE, run_shap, mrmr, SelectFdr, relief]
    #
    # # toy_example = run_toy_example()
    # # iterate_dbs(toy_example, [run_shap])
    #
    # dbs = read_dbs()
    # iterate_dbs(dbs, fs_methods)
