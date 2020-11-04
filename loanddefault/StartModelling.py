# Load system libraries
import os
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier  # GBM algorithm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score


class StartModelling:
    def __init__(self, df):
        self.path = os.getcwd() + '/FT/'
        self.lc_hist = df
        print(self.lc_hist.loan_status.unique())
        # create target variable
        self.lc_hist.loan_status = np.where(self.lc_hist.loan_status == 'Charged Off', 0, 1)
        print(self.lc_hist.loan_status.unique())
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.lc_hist.loc[:, self.lc_hist.columns != 'loan_status'],
            self.lc_hist.loan_status)

    def GBMClassifier(self):
        # param_grid = {'learning_rate': [0.1, 0.05, 0.01],
        #               'max_depth': [4, 6],
        #               'min_samples_leaf': [3, 5, 9, 17],
        #               'max_features': [1.0, 0.3, 0.1],
        #               "loss": ["deviance"]
        #               }
        # gscv = GridSearchCV(GradientBoostingClassifier(n_estimators=100),
        #                    param_grid, n_jobs=4, refit=True)
        #
        # gscv.fit(self.X_train, self.y_train)
        gscv = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2,
                                          max_depth=2, random_state=0)
        gscv.fit(self.X_train, self.y_train)
        # print(gscv.best_score_)
        print(gscv.score(self.X_test, self.y_test))
        y_pred_proba = gscv.predict_proba(self.X_test)[::, 1]
        auc = roc_auc_score(self.y_test, y_pred_proba)
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        plt.savefig(self.path + 'GBMAuc.jpg')

    def ExtraTreesClassify(self):
        # param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
        #               'max_depth': [4, 6],
        #               'min_samples_leaf': [3, 5, 9, 17],
        #               'max_features': [1.0, 0.3, 0.1]
        #               }
        # gscv = GridSearchCV(ExtraTreesClassifier(n_estimators=100),
        #                     param_grid, n_jobs=4, refit=True)
        gsev = ExtraTreesClassifier(n_estimators=20, max_features=2, criterion='entropy',
                                    max_depth=2, min_samples_split=0.1, random_state=0)
        gsev.fit(self.X_train, self.y_train)
        # print(gsev.best_score_)
        print(gsev.score(self.X_test, self.y_test))
        y_pred_proba = gsev.predict_proba(self.X_test)[::, 1]
        auc = roc_auc_score(self.y_test, y_pred_proba)
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        plt.savefig(self.path + 'ETCAuc.jpg')
