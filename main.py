import scipy.io
from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeavePOut
from BorutaShap import BorutaShap
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, precision_recall_curve, auc, \
    average_precision_score
import heapq


random_forest = RandomForestClassifier(min_samples_leaf=2, max_depth=13)
svm = SVC()
knn = KNeighborsClassifier(n_neighbors=5)
nb_classifier = GaussianNB()
logistic = LogisticRegression(random_state=0)

# models = [random_forest, svm, knn, nb_classifier, logistic]
models = [nb_classifier]


def read_dbs():
    """

    :return:
    """
    allaml_mat = scipy.io.loadmat('data/db1/ALLAML.mat')
    allaml_df = pd.DataFrame(allaml_mat['X'])
    allaml_df['y'] = allaml_mat['Y']

    # arcene_mat = scipy.io.loadmat('data/db1/arcene.mat')
    # arcene_df = pd.DataFrame(arcene_mat['X'])
    # arcene_df['y'] = arcene_mat['Y']
    #
    # Leukemia_4c_arff = arff.loadarff('data/db2/Leukemia_4c.arff')
    # Leukemia_4c_df = pd.DataFrame(Leukemia_4c_arff[0])
    #
    # Leukemia_3c_arff = arff.loadarff('data/db2/Leukemia_3c.arff')
    # Leukemia_3c_df = pd.DataFrame(Leukemia_3c_arff[0])

    # return [allaml_df, arcene_df, Leukemia_3c_df, Leukemia_4c_df]
    return [allaml_df]


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


def iterate_dbs(dbs):
    """

    :param dbs:
    :return:
    """

    for df in dbs:
        df.rename(columns=lambda x: str(x), inplace=True)
        cv_method, k = choose_method_for_cross_validation(df)
        kf = cv_method(n_splits=k)
        df.columns = [*df.columns[:-1], 'y']
        X = df.loc[:, df.columns != 'y']
        y = df['y']
        accumulated_preds = {}  # {[model]: {k: preds}}
        accumulated_y_test = {}  # {k: y_test}

        num_of_iterations = 0

        ###################
        fill_na(X)
        y.fillna(999, inplace=True)
        discretization(X)
        ###################

        for train_index, test_index in kf.split(X, y):
            print(num_of_iterations)
            X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], \
                                               y.iloc[train_index], y.iloc[test_index]

            # fill_na(X_train)
            # fill_na(X_test)
            # y_train.fillna(999, inplace=True)
            # y_test.fillna(999, inplace=True)
            #
            # discretization(X_train)
            # discretization(X_test)

            ###############################################
            run_models(X_train, y_train, X_test, y_test, 3, accumulated_preds, accumulated_y_test)
            run_models(X_train, y_train, X_test, y_test, 4, accumulated_preds, accumulated_y_test)
            ###############################################
            last_k = -1
            k_and_features_to_keep_dict = run_shap(X_train, y_train)
            # for k, features in k_and_features_to_keep_dict.items():
            #     if not last_k == k:
            #         new_X_train = X_train[features]
            #         new_X_test = X_test[features]
            #         run_models(new_X_train, y_train, new_X_test, y_test, k, accumulated_preds, accumulated_y_test)
            #     last_k = k

            num_of_iterations += 1

        evaluate_models(accumulated_preds, accumulated_y_test, num_of_iterations)


def choose_method_for_cross_validation(df):
    """

    :param df:
    :return:
    """

    df_count = len(df)

    if df_count < 50:
        return LeavePOut, 2
    elif 50 <= df_count < 100:
        return KFold, df_count
    elif 100 <= df_count < 1000:
        return KFold, 5
    elif 1000 <= df_count:
        return KFold, 10


def run_shap(X, y):
    """

    :param X:
    :param y:
    :return:
    """

    selector = BorutaShap(importance_measure='shap', classification=True)
    selector.fit(X=X, y=y, n_trials=20, sample=False, verbose=True, normalize=True)
    accepted_features_scores = selector.X_feature_import
    accepted_features_names = selector.X.columns.values
    names_and_scores_dict = dict(zip(accepted_features_names, accepted_features_scores))

    k_and_features_dict = {}
    sorted_features = sorted(names_and_scores_dict, key=names_and_scores_dict.get, reverse=True)

    # for k in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]:
    for k in [3, 4]:
        selected_features = sorted_features[:k]
        features_and_scores = {}
        for feature in selected_features:
            features_and_scores[feature] = names_and_scores_dict[feature]

        k_and_features_dict[k] = features_and_scores

    return k_and_features_dict


def run_models(X_train, y_train, X_test, y_test, k, accumulated_preds, accumulated_y_test):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """

    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_name = type(model).__name__

        if not model_name in accumulated_preds:
            accumulated_preds[model_name] = {}

        if not k in accumulated_preds[model_name]:
            accumulated_preds[model_name][k] = []

        if not k in accumulated_y_test:
            accumulated_y_test[k] = []

        accumulated_y_test[k] += y_test.tolist()
        accumulated_preds[model_name][k] += y_pred.tolist()


def evaluate_models(accumulated_preds, accumulated_y_test, num_of_iterations):
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
    print(models_scores)


if __name__ == '__main__':
    dbs = read_dbs()
    iterate_dbs(dbs)
