import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
import datetime as dt
from ltfs.train_aox2Jxw.loandefaultprediction import LoanPrediction


class LoanProcessing:
    def __init__(self):
        self.path = os.getcwd() + '/ltfs/train_aox2Jxw/'

    def readFiles(self):
        train = pd.read_csv(self.path + "train.csv")
        val = pd.read_csv(self.path + "test_bqCt9Pv.csv")
        return train, val

    def dataMunging(self, df):
        for i in range(49, 69, 1):
            df['Date.of.Birth'] = df['Date.of.Birth'].replace("-" + str(i), "-19" + str(i), regex=True)
        df['Date.of.Birth'] = pd.to_datetime(df['Date.of.Birth'])
        df['DisbursalDate'] = pd.to_datetime(df['DisbursalDate'])
        df['LoanAge'] = dt.datetime.now().year - df['DisbursalDate'].dt.year
        df['Employment.Type'].fillna(value=df['Employment.Type'].mode()[0], inplace=True)
        df['Age'] = df['DisbursalDate'].dt.year - df['Date.of.Birth'].dt.year
        df.ltv = df.ltv / 100
        df['Loan Amount'] = df.ltv * df.asset_cost
        df['undisbursedLoanAmt'] = df['Loan Amount'] - df.disbursed_amount
        df.set_index(df.DisbursalDate, drop=True, inplace=True)
        bins = [-1, 18, 300, 350, 520, 570,600, 630, 650, 680, 705, 735, 760, 804, 891]
        labels = np.arange(14,0,-1)
        df['PERFORM_CNS.SCOREBinned'] = pd.cut(df['PERFORM_CNS.SCORE'], bins=bins, labels=labels)
        # df.drop(['DisbursalDate'], inplace=True)
        print(df.head(10), df.shape)
        dfpivot = pd.pivot_table(df, index=['branch_id', 'supplier_id', 'manufacturer_id'],
                                 values=['undisbursedLoanAmt',
                                         'Loan Amount', 'Age', 'disbursed_amount', 'asset_cost', 'ltv',
                                         'PERFORM_CNS.SCORE'],
                                 aggfunc=np.mean).reset_index()
        dfpivot.rename(columns={'undisbursedLoanAmt': 'undisbursedLoanAmtBSM', 'Loan Amount': 'Loan AmountBSM',
                                'Age': 'AgeBSM', 'disbursed_amount': 'disbursed_amountBSM',
                                'asset_cost': 'asset_costBSM',
                                'ltv': 'ltvBSM', 'PERFORM_CNS.SCORE': 'PERFORM_CNS.SCOREBSM'}, inplace=True)
        dfNew = pd.merge(df, dfpivot, how="left", on=['branch_id', 'supplier_id', 'manufacturer_id'])
        print(dfNew.head(10), dfNew.shape)
        return dfNew

    def dataProcessing(self, dfT, dfV):
        features = ['disbursed_amount', 'asset_cost', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT',
                    'PRI.DISBURSED.AMOUNT', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT',
                    'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT', 'Loan Amount',
                    'undisbursedLoanAmt', 'undisbursedLoanAmtBSM','Loan AmountBSM',
                    'disbursed_amountBSM', 'asset_costBSM', 'PERFORM_CNS.SCOREBSM', 'AVERAGE.ACCT.AGE',
                    'CREDIT.HISTORY.LENGTH', 'Age', 'AgeBSM', 'branch_id', 'supplier_id',
                    'manufacturer_id', 'Current_pincode_ID']
        scaler = MinMaxScaler()
        dfT[features] = \
            scaler.fit_transform(dfT[features])
        dfV[features] = \
            scaler.transform(dfV[features])
        le = LabelEncoder()
        dfT['PERFORM_CNS.SCORE.DESCRIPTION'] = le.fit_transform(dfT['PERFORM_CNS.SCORE.DESCRIPTION'])
        dfV['PERFORM_CNS.SCORE.DESCRIPTION'] = le.transform(dfV['PERFORM_CNS.SCORE.DESCRIPTION'])
        dfT['Employment.Type'] = le.fit_transform(dfT['Employment.Type'])
        dfV['Employment.Type'] = le.transform(dfV['Employment.Type'])
        dfT.to_csv(self.path + 'dfTrain.csv')
        dfV.to_csv(self.path + 'dfVal.csv')
        return dfT, dfV

    def dataVisualisation(self, df):
        # df.describe().transpose().to_csv(self.path + "descriptivedf.csv")
        # df.set_index(df.DisbursalDate, drop=True, inplace=True)
        # dfdescpSub = df.select_dtypes(exclude=['object', 'datetime'])
        # corr = dfdescpSub.corr()
        # mask = np.zeros_like(corr, dtype=np.bool)
        # mask[np.triu_indices_from(mask)] = True
        # f, ax = plt.subplots(figsize=(80, 100))
        # cmap = sns.diverging_palette(220, 10, as_cmap=True)
        # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
        #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
        # plt.savefig(self.path + "heatmap.png")

        sns.pairplot(df[['undisbursedLoanAmtBSM', 'Loan AmountBSM','AgeBSM', 'disbursed_amountBSM',
                         'asset_costBSM', 'ltvBSM', 'PERFORM_CNS.SCOREBSM',
                         'loan_default']], size=2.5)
        plt.savefig(self.path + "pairplot.png")


if __name__ != "__main__":
    lp = LoanProcessing()
    train,val = lp.readFiles()
    train = lp.dataMunging(train)
    val = lp.dataMunging(val)
    train, val = lp.dataProcessing(train, val)
#    lp.dataVisualisation(train)
    lpd = LoanPrediction(train, val)
#    lpd.LogisticRegression()
#     lpd.RandomForest()
    # lpd.GradientBoosted()
    # lpd.Adaboost()
    # lpd.SVCModel()
    lpd.KNNclassify()
