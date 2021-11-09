import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, precision_score, precision_recall_curve, recall_score, f1_score, auc, roc_auc_score, plot_roc_curve
from sklearn.model_selection import KFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier

class BaseML():
    def __init__(self, df_csv=None):
        if df_csv != None:
            self.df = pd.read_csv(df_csv)
        else:
            pass

    def create_train_and_test_df(self, target_attribute, exclude_list = [], numtest = 0.2, rand_state = 42):
        self.X_all = self.df.drop(exclude_list, axis = 1)
        self.y_all = self.df[target_attribute]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_all, self.y_all, test_size=numtest, random_state=rand_state)


    def run_kfold(self, clf, folds = 10, name ="potato", shuffle = True):
        kf=KFold(n_splits=folds, shuffle=shuffle, random_state=42)
        outcomes = []
        precision_scores = []
        recall_scores = []
        f_scores = []
        aucs = []
        fig, ax = plt.subplots()
        table = {}
        fold = 0
        for train_index, test_index in kf.split(self.X_all):
            fold += 1
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = self.X_all.values[train_index], self.X_all.values[test_index]
            y_train, y_test = self.y_all.values[train_index], self.y_all.values[test_index]
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            predict_prob = clf.predict_proba(X_test)
            accuracy = accuracy_score(y_test, predictions)
            prec_score = precision_score(y_test, predictions, pos_label = 1)
            r_score = recall_score(y_test, predictions, pos_label = 1)
            f_score = f1_score(y_test, predictions, pos_label = 1)
            try:
                auc_score = roc_auc_score(y_test, predict_prob[:,1])
            except (ValueError, IndexError) as e:
                auc_score = 'nan'
            try:
                clf_disp = plot_roc_curve(clf, X_test, y_test, name=f"Fold {fold}", ax=ax)
            except Exception as e:
                print(f"Exception in plot_roc_curve: {e}")
            
            #plt.savefig(f"{name}_fold{fold}_test.png")
            outcomes.append(accuracy)
            precision_scores.append(prec_score)
            recall_scores.append(r_score)
            f_scores.append(f_score)
            aucs.append(auc_score)
            print(f"Fold {fold} accuracy: {accuracy}")     
        mean_outcome = np.mean(outcomes)
        mean_precision = np.mean(precision_scores)
        mean_recall = np.mean(recall_scores)
        mean_f_scores = np.mean(f_scores)
        if 'nan' in aucs: 
            mean_aucs = 'nan'
        else:
            mean_aucs = np.mean(aucs)
        table["Folds"] = list(range(1, folds+1))
        table["Folds"].append("Mean")
        table["Accuracy"] = outcomes
        table["Accuracy"].append(mean_outcome)
        table["Precision"] = precision_scores
        table["Precision"].append(mean_precision)
        table["Recall"] = recall_scores
        table["Recall"].append(mean_recall)
        table["F Score"] = f_scores
        table["F Score"].append(mean_f_scores) 
        table["AUC"] = aucs
        table["AUC"].append(mean_aucs)
        pd_table = pd.DataFrame(table)
        print(pd_table)
        pd_table.to_csv(f"{name}_table.csv")
        print(tabulate([["Mean", mean_outcome, mean_precision, mean_recall, mean_f_scores, mean_aucs]]))
        print("Mean Accuracy: {0}".format(mean_outcome))
        plt.savefig(f"{name}_AUC.png")

    def rebalance_undersampling(self, majority_val, minority_val, target_attribute):
        df_majority = self.df[self.df[target_attribute] == majority_val]
        df_minority = self.df[self.df[target_attribute] == minority_val]

        df_majority_downsampled = resample(df_majority, replace=False, n_samples=df_minority[target_attribute].size, random_state=42)
        return pd.concat([df_majority_downsampled, df_minority], axis=0)

    def rebalance_oversampling(self, majority_val, minority_val, target_attribute):
        df_majority = self.df[self.df[target_attribute] == majority_val]
        df_minority = self.df[self.df[target_attribute] == minority_val]

        df_minority_upsampled = resample(df_minority, replace=True, n_samples=df_majority[target_attribute].size, random_state=42)
        return pd.concat([df_minority_upsampled, df_majority], axis=0)

    def normalize_features(self):
        scaler = preprocessing.StandardScaler().fit(self.df)
        df_scaled = scaler.transform(self.df)
        self.df = pd.DataFrame(df_scaled, columns = self.df.columns, dtype= 'int64')
