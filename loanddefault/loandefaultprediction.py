from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

class LoanPrediction:
    def __init__(self, train, val):
        self.path = os.getcwd() + '/ltfs/train_aox2Jxw/'
        self.train = train
        self.val = val
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.train.loc[:, self.train.columns != 'loan_default'],
            self.train.loan_default)

    def LogisticRegression(self):
        features = [
                    'Loan Amount',
                    'disbursed_amount',
                    'asset_cost',
                    'ltv',
                    'Age',
                    'Employment.Type',
                    'State_ID',
                    'MobileNo_Avl_Flag',
                    'Driving_flag',
                    'PERFORM_CNS.SCORE.DESCRIPTION',
                    'PRI.NO.OF.ACCTS',
                    'Aadhar_flag',
                    'PAN_flag',
                    'VoterID_flag',
                    'Passport_flag',
                    'PRI.OVERDUE.ACCTS',
                    'PRI.CURRENT.BALANCE',
                    'PRI.SANCTIONED.AMOUNT',
                    'PRI.DISBURSED.AMOUNT',
                    'SEC.CURRENT.BALANCE',
                    'SEC.SANCTIONED.AMOUNT',
                    'SEC.DISBURSED.AMOUNT',
                    'PRIMARY.INSTAL.AMT',
                    'SEC.INSTAL.AMT',
                    'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
                    'undisbursedLoanAmt',
                    'PERFORM_CNS.SCOREBSM',
                    'Loan AmountBSM',
                    'PERFORM_CNS.SCOREBSM',
                    'asset_costBSM',
                    'disbursed_amountBSM',
                    'ltvBSM',
                    'undisbursedLoanAmtBSM',
                    'PERFORM_CNS.SCOREBinned'
                    ]
        clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
        X_train = self.X_train[features]
        X_test = self.X_test[features]
        model = self.modelbuild(X_train, X_test, clf)
        print(list(zip(list(model.coef_), features)))

    def RandomForest(self):
        parameters = {
                       'n_estimators': [1500,2000]
                     }
        n_jobs = 4
        gcv = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=n_jobs)
        features = [
            # 'disbursed_amount',
            # 'asset_cost',
            'ltv',
            # 'Employment.Type',
            'State_ID',
            # 'MobileNo_Avl_Flag',
            # 'Aadhar_flag',
            # 'PAN_flag',
            # 'VoterID_flag',
            # 'Driving_flag',
            # 'Passport_flag',
            # 'PERFORM_CNS.SCORE.DESCRIPTION',
            # 'PRI.NO.OF.ACCTS',
            'PRI.ACTIVE.ACCTS',
            'PRI.OVERDUE.ACCTS',
            'PRI.CURRENT.BALANCE',
            'PRI.SANCTIONED.AMOUNT',
            'PRI.DISBURSED.AMOUNT',
            # 'SEC.NO.OF.ACCTS',
            # 'SEC.ACTIVE.ACCTS',
            # 'SEC.OVERDUE.ACCTS',
            # 'SEC.CURRENT.BALANCE',
            # 'SEC.SANCTIONED.AMOUNT',
            # 'SEC.DISBURSED.AMOUNT',
            'PRIMARY.INSTAL.AMT',
            # 'SEC.INSTAL.AMT',
            # 'NEW.ACCTS.IN.LAST.SIX.MONTHS',
            # 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
            # 'AVERAGE.ACCT.AGE',
            'CREDIT.HISTORY.LENGTH',
            # 'NO.OF_INQUIRIES',
            'Age',
            'LoanAge',
            # 'Loan Amount',
            'undisbursedLoanAmt',
            # 'AgeBSM',
            # 'Loan AmountBSM',
            'PERFORM_CNS.SCOREBSM',
            # 'asset_costBSM',
            # 'disbursed_amountBSM',
            'ltvBSM',
            # 'undisbursedLoanAmtBSM',
            'branch_id',
            'supplier_id',
            # 'manufacturer_id',
            'Current_pincode_ID',
            'PERFORM_CNS.SCOREBinned'
        ]
        gcv.fit(self.X_train[features], self.y_train)
        print(gcv.best_params_)
        RF_clf = RandomForestClassifier(random_state=123, n_estimators=2000, criterion='entropy',
                                        class_weight={0: 0.2, 1: 5.73},min_samples_leaf=3,
                                        max_depth=8, max_features='sqrt', oob_score=True)
        X_train = self.X_train[features]
        X_test = self.X_test[features]
        val = self.val[features]
        model, self.valRF = self.modelbuild(X_train, X_test, val, RF_clf)
        Var_Importance_RF = pd.Series(data=model.feature_importances_, index=X_train.columns, name='variables')
        print(Var_Importance_RF)
        #        least_importance = Var_Importance_RF[Var_Importance_RF.values < 0.01].index
        #        X_train = X_train.drop(least_importance, axis=1).copy()
        #        X_test =  X_test.drop(least_importance, axis=1).copy()
        #        RF_clf.fit(X_train, self.y_train)

        # val_pred_proba = RF_clf.predict_proba(self.val[features])[::, 1]
        # self.y_predRF = np.where(val_pred_proba > Optthresh, 1, 0)
        self.val['loan_default'] = self.valRF
        self.val.to_csv(self.path + 'submitRF3.csv', index=False)

    def GradientBoosted(self):
        parameters = {
            "loss": ["deviance"],
            "learning_rate": [0.01, 0.025],# 0.05, 0.075, 0.1, 0.15, 0.2],
            "min_samples_split": np.linspace(0.1, 0.3, 5),
            "min_samples_leaf": np.linspace(0.1, 0.3, 5),
            "max_depth": [3, 8],
            "max_features": ["log2", "sqrt"],
            "criterion": ["friedman_mse", "mae"],
            "subsample": [0.5, 0.618],# 0.8, 0.85, 0.9, 0.95, 1.0],
            "n_estimators":  [75, 100]#[10, 20, 50,
        }
        n_jobs = 10
        gcv = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=n_jobs)
        features = [
            'disbursed_amount',
            # 'asset_cost',
            # 'ltv',
            # 'Employment.Type',
            'State_ID',
            # 'MobileNo_Avl_Flag',
            'Aadhar_flag',
            'PAN_flag',
            # 'VoterID_flag',
            # 'Driving_flag',
            # 'Passport_flag',
            'PERFORM_CNS.SCORE.DESCRIPTION',
            # 'PRI.NO.OF.ACCTS',
            # 'PRI.ACTIVE.ACCTS',
            # 'PRI.OVERDUE.ACCTS',
            'PRI.CURRENT.BALANCE',
            'PRI.SANCTIONED.AMOUNT',
            'PRI.DISBURSED.AMOUNT',
            # 'SEC.NO.OF.ACCTS',
            # 'SEC.ACTIVE.ACCTS',
            # 'SEC.OVERDUE.ACCTS',
            # 'SEC.CURRENT.BALANCE',
            # 'SEC.SANCTIONED.AMOUNT',
            # 'SEC.DISBURSED.AMOUNT',
            'PRIMARY.INSTAL.AMT',
            # 'SEC.INSTAL.AMT',
            # 'NEW.ACCTS.IN.LAST.SIX.MONTHS',
            # 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
            # 'AVERAGE.ACCT.AGE',
            # 'CREDIT.HISTORY.LENGTH',
            # 'NO.OF_INQUIRIES',
            'Age',
            'LoanAge',
            'Loan Amount',
            'undisbursedLoanAmt',
            # 'AgeBSM',
            # 'Loan AmountBSM',
            'PERFORM_CNS.SCOREBSM',
            # 'asset_costBSM',
            # 'disbursed_amountBSM',
            # 'ltvBSM',
            # 'undisbursedLoanAmtBSM',
            'branch_id',
            'supplier_id',
            # 'manufacturer_id',
            'Current_pincode_ID',
            'PERFORM_CNS.SCOREBinned'
        ]
        X_train = self.X_train[features]
        X_test = self.X_test[features]
        val = self.val[features]
        # gcv.fit(X_train, self.y_train)
        # print( gcv.best_params_)
        clf = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.001,
                                         max_depth=3, random_state=123, min_samples_split=5,
                                         min_impurity_decrease=0.1)
        model, self.valGBT = self.modelbuild(X_train, X_test, val, clf)
        print(model.score(self.X_train[features], self.y_train))
        print(model.score(self.X_test[features], self.y_test))
        Var_Importance_RF = pd.Series(data=model.feature_importances_, index=self.X_train[features].columns, name='variables')
        print(Var_Importance_RF)
        self.val['loan_default'] = self.valGBT
        self.val[['UniqueID', 'loan_default']].to_csv(self.path + 'submitgdc2.csv', index=False)

    def Adaboost(self):
        features = [
            'disbursed_amount',
            'asset_cost',
            'ltv',
            #                                'Employment.Type',
            'State_ID',
            #                                'MobileNo_Avl_Flag',
            #                                'Aadhar_flag',
            #                                'PAN_flag',
            #                                'VoterID_flag',
            #                                'Driving_flag',
            #                                'Passport_flag',
            'PERFORM_CNS.SCORE.DESCRIPTION',
            # 'PRI.NO.OF.ACCTS',
            'PRI.ACTIVE.ACCTS',
            'PRI.OVERDUE.ACCTS',
            'PRI.CURRENT.BALANCE',
            'PRI.SANCTIONED.AMOUNT',
            'PRI.DISBURSED.AMOUNT',
            #                                'SEC.NO.OF.ACCTS',
            #                                'SEC.ACTIVE.ACCTS',
            #                                'SEC.OVERDUE.ACCTS',
            #                                'SEC.CURRENT.BALANCE',
            #                                'SEC.SANCTIONED.AMOUNT',
            #                                'SEC.DISBURSED.AMOUNT',
            'PRIMARY.INSTAL.AMT',
            # 'SEC.INSTAL.AMT',
            # 'NEW.ACCTS.IN.LAST.SIX.MONTHS',
            # 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
            # 'AVERAGE.ACCT.AGE',
            'CREDIT.HISTORY.LENGTH',
            # 'NO.OF_INQUIRIES',
            'Age',
            'LoanAge',
            'Loan Amount',
            'undisbursedLoanAmt',
            # 'AgeBSM',
            # 'Loan AmountBSM',
            'PERFORM_CNS.SCOREBSM',
            # 'asset_costBSM',
            # 'disbursed_amountBSM',
            # 'ltvBSM',
            # 'undisbursedLoanAmtBSM',
            'branch_id',
            'supplier_id',
            # 'manufacturer_id',
            'Current_pincode_ID',
            'PERFORM_CNS.SCOREBinned'
        ]
        X_train = self.X_train[features]
        X_test = self.X_test[features]
        RF_clf = RandomForestClassifier(random_state=123, n_estimators=200, criterion='entropy',
                                        class_weight={0: 0.2, 1: 0.8}, min_samples_leaf=5,
                                        max_depth=10, max_features='sqrt', oob_score=True)
        abclf = AdaBoostClassifier(RF_clf, n_estimators=200, learning_rate=0.01, random_state=123)
        model, val_pred = self.modelbuild(X_train, X_test, abclf)

    def SVCModel(self):
        svm = SVC(kernel='rbf', degree=3, random_state=123, gamma="auto", C=5.0,
                  class_weight={0: 0.2, 1: 0.8}, probability=False)
        features = [
            'disbursed_amount',
            'asset_cost',
            # 'ltv',
            'Employment.Type',
            'State_ID',
            # 'MobileNo_Avl_Flag',
            # 'Aadhar_flag',
            'PAN_flag',
            # 'VoterID_flag',
            # 'Driving_flag',
            # 'Passport_flag',
            'PERFORM_CNS.SCORE.DESCRIPTION',
            'PRI.NO.OF.ACCTS',
            'PRI.ACTIVE.ACCTS',
            'PRI.OVERDUE.ACCTS',
            'PRI.CURRENT.BALANCE',
            'PRI.SANCTIONED.AMOUNT',
            'PRI.DISBURSED.AMOUNT',
            # 'SEC.NO.OF.ACCTS',
            # 'SEC.ACTIVE.ACCTS',
            # 'SEC.OVERDUE.ACCTS',
            # 'SEC.CURRENT.BALANCE',
            # 'SEC.SANCTIONED.AMOUNT',
            # 'SEC.DISBURSED.AMOUNT',
            # 'PRIMARY.INSTAL.AMT',
            # 'SEC.INSTAL.AMT',
            'NEW.ACCTS.IN.LAST.SIX.MONTHS',
            # 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
            'AVERAGE.ACCT.AGE',
            'CREDIT.HISTORY.LENGTH',
            # 'NO.OF_INQUIRIES',
            'Age',
            'LoanAge',
            'Loan Amount',
            # 'undisbursedLoanAmt',
            # 'AgeBSM',
            # 'Loan AmountBSM',
            # 'PERFORM_CNS.SCOREBSM',
            # 'asset_costBSM',
            # 'disbursed_amountBSM',
            # 'ltvBSM',
            # 'undisbursedLoanAmtBSM',
            # 'branch_id',
            'supplier_id',
            # 'manufacturer_id',
            'Current_pincode_ID',
            'PERFORM_CNS.SCOREBinned'
        ]
        X_train = self.X_train[features]
        X_test = self.X_test[features]
        val = self.val[features]
        model, self.valSVC = self.modelbuild(X_train, X_test, val, svm)
        self.val['loan_default'] = self.valSVC
        self.val[['UniqueID', 'loan_default']].to_csv(self.path + 'submitRF2.csv', index=False)

    def KNNclassify(self):
        knn = KNeighborsClassifier(n_neighbors=3)
        features = [
            'disbursed_amount',
            # 'asset_cost',
            'ltv',
            # 'Employment.Type',
            'State_ID',
            # 'MobileNo_Avl_Flag',
            # 'Aadhar_flag',
            'PAN_flag',
            # 'VoterID_flag',
            # 'Driving_flag',
            # 'Passport_flag',
            # 'PERFORM_CNS.SCORE.DESCRIPTION',
            # 'PRI.NO.OF.ACCTS',
            # 'PRI.ACTIVE.ACCTS',
            'PRI.OVERDUE.ACCTS',
            'PRI.CURRENT.BALANCE',
            'PRI.SANCTIONED.AMOUNT',
            'PRI.DISBURSED.AMOUNT',
            # 'SEC.NO.OF.ACCTS',
            # 'SEC.ACTIVE.ACCTS',
            # 'SEC.OVERDUE.ACCTS',
            # 'SEC.CURRENT.BALANCE',
            # 'SEC.SANCTIONED.AMOUNT',
            # 'SEC.DISBURSED.AMOUNT',
            # 'PRIMARY.INSTAL.AMT',
            # 'SEC.INSTAL.AMT',
            'NEW.ACCTS.IN.LAST.SIX.MONTHS',
            # 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
            # 'AVERAGE.ACCT.AGE',
            'CREDIT.HISTORY.LENGTH',
            # 'NO.OF_INQUIRIES',
            'Age',
            'LoanAge',
            # 'Loan Amount',
            # 'undisbursedLoanAmt',
            # 'AgeBSM',
            # 'Loan AmountBSM',
            # 'PERFORM_CNS.SCOREBSM',
            # 'asset_costBSM',
            # 'disbursed_amountBSM',
            # 'ltvBSM',
            # 'undisbursedLoanAmtBSM',
            # 'branch_id',
            'supplier_id',
            # 'manufacturer_id',
            'Current_pincode_ID',
            'PERFORM_CNS.SCOREBinned'
        ]
        X_train = self.X_train[features]
        X_test = self.X_test[features]
        val = self.val[features]
        model, self.valKNN = self.modelbuild(X_train, X_test, val, knn)
        print(model)
        self.val['loan_default'] = self.valKNN
        self.val[['UniqueID', 'loan_default']].to_csv(self.path + 'submitKNN1.csv', index=False)

    def modelbuild(self, train, test, val, model ):
        model.fit(train, self.y_train)
        print(model.score(train, self.y_train))
        print(model.score(test, self.y_test))
        y_pred_proba = model.predict_proba(test)[::, 1]
        auc = metrics.roc_auc_score(self.y_test, y_pred_proba)
        fpr, tpr, _ = metrics.roc_curve(self.y_test, y_pred_proba)
        plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        plt.show()
        precision, recall, thresholds = metrics.precision_recall_curve(self.y_test, y_pred_proba)
        thresholds = np.append(thresholds, 1)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        metricsDF = pd.DataFrame(
            {'precision': precision, 'recall': recall, 'thresholds': thresholds, 'f1_scores': f1_scores})
        Optthresh = metricsDF.loc[metricsDF['f1_scores'] == metricsDF['f1_scores'].max(), 'thresholds'].iloc[0]
        y_pred1 = np.where(y_pred_proba > Optthresh, 1, 0)
        print(metrics.accuracy_score(self.y_test, y_pred1))
        print(metrics.confusion_matrix(self.y_test, y_pred1))
        print(Optthresh, metrics.f1_score(self.y_test, y_pred1))
        y_pred = model.predict(test)
        print("PredAccuracy:", metrics.accuracy_score(self.y_test, y_pred))
        Confusion_Mat_test_RF = metrics.confusion_matrix(self.y_test, y_pred)
        print(Confusion_Mat_test_RF), print(metrics.f1_score(self.y_test, y_pred))
        aucp = metrics.roc_auc_score(self.y_test, y_pred)
        fpr1, tpr1, _ = metrics.roc_curve(self.y_test, y_pred)
        plt.plot(fpr1, tpr1, label="data 2, auc=" + str(aucp))
        plt.legend(loc=4)
        plt.show()

        val_pred_proba = model.predict_proba(val)[::, 1]
        val_pred_proba = np.where(val_pred_proba > Optthresh, 1, 0)
        return model, val_pred_proba

    def ensembleModel(self):
        self.val['loan_default'] = 0.25*self.valRF + 0.5*self.valGBT + 0.25*self.valSVC
        self.val[['UniqueID', 'loan_default']].to_csv(self.path + 'submitens1.csv')