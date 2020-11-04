from sklearn.feature_selection import RFECV  # , RFE
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.fixes import signature
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

sns_colors = sns.color_palette('colorblind')

from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)

'''
ForeClosureModelling: Class for feature selection and to try various algorithms for training and prediction
Used Tensorflow Logistic/Boosted trees, sklearn for Logistic Regression
The final model used is BoostedTreesClassifiertf:
Model Metrics:
accuracy: 0.73,  auc_precision_recall: 0.78, auc: 0.72
average_loss: 0.69, global_step: 100.00, loss: 0.12, precision: 0.79 
'''


class ForeclosureModelling:

    def __init__(self, X_train, y_train, X_test, y_test, X_val, y_val):
        self.filePath = os.getcwd() + '/foreclosureprediction/data/'
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val

    # Function to know the optimal features to use
    def featureSelection(self):
        # ['AGE', 'AGEPATTERN', 'BASE_RATE@EMIRECEIPTDATE', 'contractualpay', 'CURR_BASERATE', 'CURRENT_ROI', 'custmailfreq', 'EMI_OS_AMOUNT', 'EXCESS_ADJUSTED_AMT', 'FOIR', 'LOANAGE', 'loanprepay', 'NET_DISBURSED_AMT',
        # 'NET_LTV', 'NET_RECEIVABLE', 'NO_OF_DEPENDENT', 'ORIGNAL_ROI', 'PRE_EMI_DUEAMT', 'PRODUCTEncode', 'TENURECHANGE', 'volparpay']
        rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
        rfecv.fit(self.X_train[['AGE', 'AGEPATTERN', 'BASE_RATE@EMIRECEIPTDATE', 'contractualpay', 'CURR_BASERATE',
                                'CURRENT_ROI',
                                'custmailfreq', 'default', 'EMI_DUEAMT', 'EMI_OS_AMOUNT', 'EMI_RECEIVED_AMT',
                                'EXCESS_ADJUSTED_AMT',
                                'FOIR', 'GROSSINCOMEIMPUTED', 'LOANAGE', 'LOAN_AMT', 'loanprepay', 'loanrefinance',
                                'MONTHOPENING',
                                'NET_DISBURSED_AMT', 'NET_LTV', 'NET_RECEIVABLE', 'NO_OF_DEPENDENT', 'NPALoan60-90',
                                'NPALoanGT90',
                                'ORIG_BASERATE', 'ORIGNAL_ROI', 'OUTSTANDING_PRINCIPAL', 'PAID_INTEREST',
                                'PAID_PRINCIPAL',
                                'PRE_EMI_DUEAMT', 'PRODUCTEncode', 'TENURECHANGE', 'volparpay']]
                  , self.y_train)
        # rfe = rfe.fit(self.X, self.y)
        print("Optimal number of features: %d" % rfecv.n_features_)
        print(
            'Selected features: %s' % list(self.X_train[
                                               ['AGE', 'AGEPATTERN', 'BASE_RATE@EMIRECEIPTDATE', 'contractualpay',
                                                'CURR_BASERATE', 'CURRENT_ROI',
                                                'custmailfreq', 'default', 'EMI_DUEAMT', 'EMI_OS_AMOUNT',
                                                'EMI_RECEIVED_AMT', 'EXCESS_ADJUSTED_AMT',
                                                'FOIR', 'GROSSINCOMEIMPUTED', 'LOANAGE', 'LOAN_AMT', 'loanprepay',
                                                'loanrefinance', 'MONTHOPENING',
                                                'NET_DISBURSED_AMT', 'NET_LTV', 'NET_RECEIVABLE', 'NO_OF_DEPENDENT',
                                                'NPALoan60-90', 'NPALoanGT90',
                                                'ORIG_BASERATE', 'ORIGNAL_ROI', 'OUTSTANDING_PRINCIPAL',
                                                'PAID_INTEREST', 'PAID_PRINCIPAL',
                                                'PRE_EMI_DUEAMT', 'PRODUCTEncode', 'TENURECHANGE',
                                                'volparpay']].columns[rfecv.support_]))

        # Plot number of features VS. cross-validation scores
        plt.figure(figsize=(10, 6))
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()

    def dropcol_importances(self):
        rf = RandomForestClassifier(n_estimators=1000, max_depth=20, random_state=0, oob_score=True)
        rf_ = clone(rf)
        rf_.random_state = 999
        X_train = self.X_train[['AGE', 'BASE_RATE@EMIRECEIPTDATE', 'contractualpay', 'CURR_BASERATE', 'CURRENT_ROI',
                                'custmailfreq', 'default', 'EMI_OS_AMOUNT', 'EXCESS_ADJUSTED_AMT',
                                'FOIR', 'GROSSINCOMEIMPUTED', 'LOANAGE', 'LOAN_AMT', 'loanprepay', 'loanrefinance',
                                'MONTHOPENING',
                                'NET_DISBURSED_AMT', 'NET_LTV', 'NET_RECEIVABLE', 'NO_OF_DEPENDENT', 'NPALoan60-90',
                                'NPALoanGT90',
                                'ORIG_BASERATE', 'ORIGNAL_ROI', 'PAID_INTEREST', 'PAID_PRINCIPAL',
                                'PRE_EMI_DUEAMT', 'PRODUCTEncode', 'TENURECHANGE', 'volparpay']]
        # 'EMI_DUEAMT', 'EMI_RECEIVED_AMT', 'OUTSTANDING_PRINCIPAL','AGEPATTERN',
        rf_.fit(X_train, self.y_train)
        baseline = rf_.oob_score_
        imp = []
        for col in X_train.columns:
            X = X_train.drop(col, axis=1)
            rf_ = clone(rf)
            rf_.random_state = 999
            rf_.fit(X, self.y_train)
            o = rf_.oob_score_
            imp.append(baseline - o)
        imp = np.array(imp)
        I = pd.DataFrame(
            data={'Feature': X_train.columns,
                  'Importance': imp})
        I = I.set_index('Feature')
        I = I.sort_values('Importance', ascending=True)
        return I

    # Used ScikitLearn :Logistic Regression/ OneVsRest classifier
    #https://www.kaggle.com/qianchao/smote-with-imbalance-data, https://www.kaggle.com/hatone/gradientboostingclassifier-with-gridsearchcv, https://www.kaggle.com/ccourtot/sklearn-adaboostclassifier
    def logisticRegressionsk(self):
#'EXCESS_ADJUSTED_AMT','NET_DISBURSED_AMT','NET_RECEIVABLE(0.67)','PRE_EMI_DUEAMT0.677,900,0.2',
# class_weight={0:0.1,1:0.9}auc0.68,accuracy0.74, w/o weights auc .68,accuracy0.99
#,
        clf = LogisticRegression(random_state=0, multi_class='multinomial', C=0.001, solver='newton-cg',
                                 max_iter=1200, class_weight={0: 0.1, 1: 0.9})
        x_train = self.X_train[
            ['AGE', 'BASE_RATE@EMIRECEIPTDATE', 'contractualpay', 'CURR_BASERATE', 'CURRENT_ROI',
             'custmailfreq', 'EMI_OS_AMOUNT',  'FOIR', 'LOANAGE', 'loanprepay',
             'NET_LTV',  'NO_OF_DEPENDENT', 'ORIGNAL_ROI','TENURECHANGE',
             'PRODUCTEncode']]
        x_test = self.X_test[
            ['AGE', 'BASE_RATE@EMIRECEIPTDATE', 'contractualpay', 'CURR_BASERATE', 'CURRENT_ROI',
             'custmailfreq', 'EMI_OS_AMOUNT', 'FOIR', 'LOANAGE', 'loanprepay',
             'NET_LTV', 'NO_OF_DEPENDENT', 'ORIGNAL_ROI','TENURECHANGE',
              'PRODUCTEncode']]
        model = clf.fit(x_train, self.y_train)
        print(model.coef_)
        y_pred = clf.predict(x_test)

        print("Accuracy:", metrics.accuracy_score(self.y_test, y_pred))
        y_pred_proba = clf.predict_proba(x_test)[::, 1]
        with open(self.filePath + 'predictionsprob.txt', 'wb') as pred:
            pickle.dump(y_pred_proba, pred)
        fpr, tpr, _ = metrics.roc_curve(self.y_test, y_pred_proba)
        auc = metrics.roc_auc_score(self.y_test, y_pred_proba)
        print('auc:', auc)
        print(metrics.precision_score(self.y_test, y_pred))
        # plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        # plt.legend(loc=4)
        # plt.show()

        average_precision = metrics.average_precision_score(self.y_test, y_pred_proba)
        print('Average precision-recall score: {0:0.2f}'.format(average_precision))
        precision, recall, _ = metrics.precision_recall_curve(self.y_test, y_pred_proba)

        # # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        # step_kwargs = ({'step': 'post'}
        #                if 'step' in signature(plt.fill_between).parameters
        #                else {})
        # plt.step(recall, precision, color='b', alpha=0.2,
        #          where='post')
        # plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        #
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        # plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        #     average_precision))
        pr_auc = metrics.auc(recall, precision)

        plt.title("Precision-Recall vs Threshold Chart")
        plt.plot(_, precision[: -1], "b--", label="Precision")
        plt.plot(_, recall[: -1], "r--", label="Recall")
        plt.ylabel("Precision, Recall")
        plt.xlabel("Threshold")
        plt.legend(loc="lower left")
        plt.ylim([0,1])


        # classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True,
        #                                      random_state=np.random.RandomState(0)))
        # y_score = classifier.fit(
        #     self.X_train[
        #         ['AGE', 'AGEPATTERN', 'BASE_RATE@EMIRECEIPTDATE', 'contractualpay', 'CURR_BASERATE', 'CURRENT_ROI',
        #          'custmailfreq', 'EMI_OS_AMOUNT', 'EXCESS_ADJUSTED_AMT', 'FOIR', 'LOANAGE', 'loanprepay',
        #          'NET_DISBURSED_AMT', 'NET_LTV', 'NET_RECEIVABLE', 'NO_OF_DEPENDENT', 'ORIGNAL_ROI',
        #          'PRE_EMI_DUEAMT', 'PRODUCTEncode', 'TENURECHANGE', 'volparpay']], self.y_train). \
        #     decision_function(
        #     self.X_test[
        #         ['AGE', 'AGEPATTERN', 'BASE_RATE@EMIRECEIPTDATE', 'contractualpay', 'CURR_BASERATE', 'CURRENT_ROI',
        #          'custmailfreq', 'EMI_OS_AMOUNT', 'EXCESS_ADJUSTED_AMT', 'FOIR', 'LOANAGE', 'loanprepay',
        #          'NET_DISBURSED_AMT', 'NET_LTV', 'NET_RECEIVABLE', 'NO_OF_DEPENDENT', 'ORIGNAL_ROI',
        #          'PRE_EMI_DUEAMT', 'PRODUCTEncode', 'TENURECHANGE', 'volparpay']])
        # with open(self.filePath + 'predictionsprob.txt', 'wb') as pred:
        #     pickle.dump(y_score, pred)
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # for i in range(self.n_classes):
        #     fpr[i], tpr[i], _ = metrics.roc_curve(self.y_test[:, i], y_score[:, i])
        #     roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        # fpr["micro"], tpr["micro"], _ = metrics.roc_curve(self.y_test.ravel(), y_score.ravel())
        # roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
        # plt.figure()
        # lw = 2
        # plt.plot(fpr[1], tpr[1], color='darkorange',
        #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic example')
        # plt.legend(loc="lower right")
        # plt.show()

    def boostedTrees(self):
        x_train = self.X_train[
            ['AGE', 'ASSESTVALUE', 'BASE_RATE@EMIRECEIPTDATE', 'BASE_RATE@INCEPTION',
             'contractualpay', 'CURR_BASERATE', 'CURRENT_ROI', 'custmailfreq','default',  'EMI_DUEAMT', 'EMI_OS_AMOUNT',
             'EMI_RECEIVED_AMT', 'EXCESS_AVAILABLE', 'FOIR', 'GROSSINCOMEIMPUTED', 'LOANAGE', 'LOAN_AMT',
             'loanprepay', 'loanrefinance', 'MONTHOPENING', 'NO_OF_DEPENDENT',
             'ORIG_BASERATE', 'ORIGNAL_ROI', 'ORIGNAL_TENOR', 'OUTSTANDING_PRINCIPAL', 'PAID_INTEREST',
             'PAID_PRINCIPAL', 'PRE_EMI_DUEAMT', 'PRODUCTEncode', 'TENURECHANGE', 'volparpay']]
        #'AGEPATTERN','COMPLETED_TENURE','EXCESS_ADJUSTED_AMT','NPALoan30-60', 'NPALoan60-90','NPALoanGT90','NET_LTV',
        # 'NET_RECEIVABLE',
        x_test = self.X_test[
            ['AGE',  'ASSESTVALUE', 'BASE_RATE@EMIRECEIPTDATE', 'BASE_RATE@INCEPTION',
              'contractualpay', 'CURR_BASERATE','CURRENT_ROI',  'custmailfreq', 'default','EMI_DUEAMT', 'EMI_OS_AMOUNT',
             'EMI_RECEIVED_AMT', 'EXCESS_AVAILABLE', 'FOIR', 'GROSSINCOMEIMPUTED', 'LOANAGE', 'LOAN_AMT',
             'loanprepay', 'loanrefinance','MONTHOPENING',   'NO_OF_DEPENDENT',
              'ORIG_BASERATE', 'ORIGNAL_ROI', 'ORIGNAL_TENOR', 'OUTSTANDING_PRINCIPAL', 'PAID_INTEREST',
             'PAID_PRINCIPAL', 'PRE_EMI_DUEAMT', 'PRODUCTEncode', 'TENURECHANGE', 'volparpay']]
        gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.03, n_estimators=2000, max_depth=40,
                                                 min_samples_split=4600, min_samples_leaf=50, subsample=0.8,
                                                 random_state=10, max_features=1)
        gbm_tuned_1.fit(x_train, self.y_train)
        dtrain_predictions = gbm_tuned_1.predict(x_test)
        dtrain_predprob = gbm_tuned_1.predict_proba(x_test)[:, 1]
        print("Accuracy : %.4g" % metrics.accuracy_score(self.y_test, dtrain_predictions))
        print("AUC Score (Train): %f" % metrics.roc_auc_score(self.y_test, dtrain_predprob))
        print('Average precision-recall score: {0:0.2f}'.format(metrics.average_precision_score(self.y_test, dtrain_predprob)))

    def adaBoost(self):
        x_train = self.X_train[
            ['AGE', 'ASSESTVALUE', 'BASE_RATE@EMIRECEIPTDATE', 'BASE_RATE@INCEPTION',
             'contractualpay', 'CURR_BASERATE', 'CURRENT_ROI', 'custmailfreq','default',  'EMI_DUEAMT', 'EMI_OS_AMOUNT',
             'EMI_RECEIVED_AMT', 'EXCESS_AVAILABLE', 'FOIR', 'GROSSINCOMEIMPUTED', 'LOANAGE', 'LOAN_AMT',
             'loanprepay', 'loanrefinance', 'MONTHOPENING', 'NO_OF_DEPENDENT',
             'ORIG_BASERATE', 'ORIGNAL_ROI', 'ORIGNAL_TENOR', 'OUTSTANDING_PRINCIPAL', 'PAID_INTEREST',
             'PAID_PRINCIPAL', 'PRE_EMI_DUEAMT', 'PRODUCTEncode', 'TENURECHANGE', 'volparpay']]
        #'AGEPATTERN','COMPLETED_TENURE','EXCESS_ADJUSTED_AMT','NPALoan30-60', 'NPALoan60-90','NPALoanGT90','NET_LTV',
        # 'NET_RECEIVABLE',
        x_test = self.X_test[
            ['AGE',  'ASSESTVALUE', 'BASE_RATE@EMIRECEIPTDATE', 'BASE_RATE@INCEPTION',
              'contractualpay', 'CURR_BASERATE','CURRENT_ROI',  'custmailfreq', 'default','EMI_DUEAMT', 'EMI_OS_AMOUNT',
             'EMI_RECEIVED_AMT', 'EXCESS_AVAILABLE', 'FOIR', 'GROSSINCOMEIMPUTED', 'LOANAGE', 'LOAN_AMT',
             'loanprepay', 'loanrefinance','MONTHOPENING',   'NO_OF_DEPENDENT',
              'ORIG_BASERATE', 'ORIGNAL_ROI', 'ORIGNAL_TENOR', 'OUTSTANDING_PRINCIPAL', 'PAID_INTEREST',
             'PAID_PRINCIPAL', 'PRE_EMI_DUEAMT', 'PRODUCTEncode', 'TENURECHANGE', 'volparpay']]

        # svc = SVC(probability=True, kernel='linear', class_weight={0:0.2,1:0.8})
        rf = RandomForestClassifier(n_estimators=1000, max_depth=20, random_state=0, oob_score=True)
        clf = LogisticRegression(random_state=0, multi_class='multinomial', C=0.001, solver='newton-cg',
                                 max_iter=1200)
        dtc = DecisionTreeClassifier(max_depth=20)
        abc = AdaBoostClassifier(n_estimators=2000, base_estimator=rf, learning_rate=0.03)
        abc.fit(x_train, self.y_train)
        dtest_predictions = abc.predict(x_test)
        dtest_predprob = abc.predict_proba(x_test)[:, 1]
        print("Accuracy : %.4g" % metrics.accuracy_score(self.y_test, dtest_predictions))
        print("AUC Score (Train): %f" % metrics.roc_auc_score(self.y_test, dtest_predprob))
        print('Average precision-recall score: {0:0.2f}'.format(metrics.average_precision_score(self.y_test, dtest_predprob)))