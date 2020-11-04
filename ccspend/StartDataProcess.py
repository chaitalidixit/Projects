import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pylab as plt
from StartModelling import StartModelling


class DataProcessing:
    def __init__(self):
        self.filePath = os.getcwd()

    def read_files(self):
        train_df = pd.read_csv(self.filePath + '/train.csv')
        test_df = pd.read_csv(self.filePath + '/test_9K3DBWQ.csv')
        return train_df, test_df

    def data_processing(self, df):
        # Handling NAs, since dataset is small performed exploration in Excel
        df['Tot_Investment'] = df[['investment_1', 'investment_2', 'investment_3', 'investment_4']].sum(
            axis=1)
        df['Tot_Loans'] = df[['personal_loan_active', 'vehicle_loan_active', 'personal_loan_closed',
                              'vehicle_loan_closed']].sum(axis=1)
        df.age = np.where(df.age > 90, df.age.mean(), df.age)
        desc = pd.DataFrame(df.describe().T)
        desc.to_csv(self.filePath + '/descriptive.csv')
        df.drop(columns=['loan_enq', 'id', 'investment_1', 'investment_2', 'investment_3', 'investment_4',
                         'personal_loan_active', 'vehicle_loan_active', 'personal_loan_closed',
                         'vehicle_loan_closed'],
                inplace=True)
        # df.fillna(value=df.mean(), inplace=True)
        return df

    def data_featengg(self, df):
        df['CCAvgPerSpendApr'] = df['cc_cons_apr'] / df['cc_count_apr']
        df['DCAvgPerSpendApr'] = df['dc_cons_apr'] / df['dc_count_apr']
        df['CCAvgPerSpendMay'] = df['cc_cons_may'] / df['cc_count_may']
        df['DCAvgPerSpendMay'] = df['dc_cons_may'] / df['dc_count_may']
        df['CCAvgPerSpendJun'] = df['cc_cons_jun'] / df['cc_count_jun']
        df['DCAvgPerSpendJun'] = df['dc_cons_jun'] / df['dc_count_jun']
        df['debt_apr'] = np.where(np.subtract(df['credit_amount_apr'], df['debit_count_apr']) > 0, 0, 1)
        df['debt_may'] = np.where(np.subtract(df['credit_amount_may'], df['debit_count_may']) > 0, 0, 1)
        df['debt_jun'] = np.where(np.subtract(df['credit_amount_jun'], df['debit_count_jun']) > 0, 0, 1)
        # Label Encoding for Gender, Account, M=1, saving=1
        df.fillna(value=df.mean(), inplace=True)
        grp_region = df.iloc[:, 2:].groupby('region_code', as_index=False).agg('mean')
        grp_age = df.iloc[:, 2:].groupby('age', as_index=False).agg('mean')
        grp_acct = df.groupby('account_type', as_index=False).agg('mean')
        df = pd.merge(df, grp_region, how='left', on='region_code', suffixes=('', '_meanR'))
        df = pd.merge(df, grp_age, how='left', on='age', suffixes=('', '_meanAg'))
        df = pd.merge(df, grp_acct, how='left', on='account_type', suffixes=('', '_meanAct'))
        mapping = {'M': 1, 'F': 0, 'saving': 1, 'current': 0}
        df['gender'] = df['gender'].map(mapping)
        df['account_type'] = df['account_type'].map(mapping)

        return df

    def data_exploration(self, df):
        for col in df.columns[2:]:
            self.univariate_plot(df, col, 0)
        # for col in df.columns[2:]:
        #     self.univariate_plot(df, col, 1)

    def univariate_plot(self, df, col, var_type, hue=None):
        if var_type == 0:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 8))
            ax.set_title("Distribution Plot of " + col)
            sns.distplot(df[col], ax=ax)
            plt.savefig(self.filePath + '/visuals/uni_analysis_' + col + '.jpg')
        if var_type == 1:
            grid = sns.FacetGrid(df, row="gender", col="account_type", margin_titles=True)
            grid.map(plt.hist, col)
            plt.savefig(self.filePath + '/visuals/uni_facet_' + col + '.jpg')
        else:
            exit


if __name__ == '__main__':
    am = DataProcessing()
    train, val = am.read_files()
    train = am.data_processing(train)
    val = am.data_processing(val)
    # am.data_exploration(train)
    train = am.data_featengg(train)
    val = am.data_featengg(val)
    train1 = pd.read_csv(os.getcwd() + '/train1.csv')
    val1 = pd.read_csv(os.getcwd() + '/test1.csv')
    sm = StartModelling(train1, val1)
    # sm.rfmodel()
    predictions = sm.rfmodel()
    sm.svrmodel()


