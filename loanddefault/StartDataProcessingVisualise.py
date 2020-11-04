# Load System Libraries
import pandas as pd
import numpy as np
import os
import datetime as dt
import seaborn as sns
import matplotlib.pylab as plt
# Load App Library
# from FT.StartModelling import StartModelling

class StartDPV:
    def __init__(self):
        # self.path = os.getcwd() + '../FT/'
        self.path = 'C:\\Users\\Chaitali\\Documents\\python\\analytics_data\\FT\\'

    def read_files(self):
        lc_hist = pd.read_csv(self.path + '/lc_historical.csv', parse_dates=['earliest_cr_line', 'issue_d'],
                              infer_datetime_format=True)
        print(lc_hist.head(10))
        print(lc_hist.dtypes)
        print('preprocessing null perc', lc_hist.isnull().mean() * 100)
        # removing variables with 100% NAs, imputing all values for a variable isnt helpful for ML
        lc_hist.drop(['all_util', 'inq_last_12m'], axis=1, inplace=True)
        # removing col id since it has all unique values and hence wont help with feat. engg.
        lc_hist.drop(['id'], axis=1, inplace=True)
        return lc_hist

    def data_preprocessing(self, lc_hist):
        # convert %s to ratio values
        lc_hist['revol_util'] = pd.Series(lc_hist.revol_util).str.replace('%', '').astype(float)
        lc_hist['percent_bc_gt_75'] = lc_hist.percent_bc_gt_75 / 100

        # impute NAs
        lc_hist.replace('n/a', np.nan, inplace=True)
        lc_hist.dropna(subset=['revol_util'], inplace=True)
        lc_hist['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
        lc_hist['emp_length'] = lc_hist['emp_length'].astype(float)
        lc_hist.emp_length.fillna(value=lc_hist.emp_length.median(), inplace=True)
        lc_hist.avg_cur_bal.fillna(value=lc_hist.avg_cur_bal.mean(), inplace=True)
        lc_hist.acc_open_past_24mths.fillna(value=lc_hist.acc_open_past_24mths.median(), inplace=True)
        lc_hist.percent_bc_gt_75.fillna(value=lc_hist.percent_bc_gt_75.mean(), inplace=True)
        lc_hist.bc_util.fillna(value=lc_hist.bc_util.mean(), inplace=True)
        print('post process null perc ', lc_hist.isnull().mean() * 100)

        # Feature Engineering
        lc_hist['debt_minus_mortgage'] = lc_hist.dti * (lc_hist.annual_inc / 12.0)
        lc_hist['length_cr_line_yrs'] = dt.datetime.today() - df.earliest_cr_line
        lc_hist['length_cr_line_yrs'] = lc_hist['length_cr_line_yrs'].dt.days / 365
        lc_hist['loan_amnt_to_anninc_ratio'] = lc_hist['loan_amnt'] / lc_hist['annual_inc']
        # bin loan amount
        bins = [0, 5000, 10000, 15000, 20000, 25000, 40000]
        binname = ['0-5000', '5000-10000', '10000-15000', '15000-20000', '20000-25000', '25000 and above']
        lc_hist['loan_amnt_binned'] = pd.cut(lc_hist['loan_amnt'], bins, labels=binname)
        # Remove Outliers and bin Annual Income
        q = lc_hist["annual_inc"].quantile(0.995)
        lc_hist = lc_hist[lc_hist["annual_inc"] < q]
        bins = [0, 25000, 50000, 75000, 100000, 1000000]
        binname = ['0-25000', '25000-50000', '50000-75000', '75000-100000', '100000 and above']
        lc_hist['annual_inc_binned'] = pd.cut(lc_hist['annual_inc'], bins, labels=binname)
        # bin FICO score
        bins = [600, 630, 650, 680, 705, 735, 760, 804, 891]
        binname = ['600-630', '630-650', '650-680', '680-705', '705-735', '735-760', '760-804', '804 and above']
        lc_hist['fico_range_low_binned'] = pd.cut(lc_hist['fico_range_low'], bins, labels=binname)
        return lc_hist

    def univariate_plot(self, lc_hist, col, var_type, hue=None):
        if var_type == 0:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 8))
            ax.set_title("Distribution Plot")
            sns.distplot(lc_hist[col], ax=ax)
        if var_type == 1:
            temp = pd.Series(data=hue)
            fig, ax = plt.subplots()
            width = len(lc_hist[col].unique()) + 6 + 4 * len(temp.unique())
            fig.set_size_inches(width, 7)
            ax = sns.countplot(data=lc_hist, x=col, order=lc_hist[col].value_counts().index, hue=hue)
            if len(temp.unique()) > 0:
                for p in ax.patches:
                    ax.annotate('{:1.1f}%'.format((p.get_height() * 100) / float(len(lc_hist))),
                                (p.get_x() + 0.05, p.get_height() + 20))
            else:
                for p in ax.patches:
                    ax.annotate(p.get_height(), (p.get_x() + 0.32, p.get_height() + 20))
            del temp
        else:
            exit

        plt.savefig(self.path + 'uni_analysis_' + col + '.jpg')

    def group_summary(self, lc_hist, col):
        grp_df = pd.crosstab(lc_hist[col], lc_hist['loan_status'], margins=True)
        grp_df['Probability_Charged Off'] = round((grp_df['Charged Off'] / grp_df['All']), 3)
        grp_df = grp_df[0:-1]
        return grp_df

    def bivariate_plot(self, lc_hist, col, stacked=True):
        # get data_frame from group_summary function
        grp_df = self.group_summary(lc_hist, col)
        line_plt = grp_df[['Probability_Charged Off']]
        bar_plt = grp_df.iloc[:, 0:2]
        ax = line_plt.plot(figsize=(20, 8), marker='o', color='b')
        ax2 = bar_plt.plot(kind='bar', ax=ax, rot=1, secondary_y=True, stacked=stacked)
        ax.set_title(lc_hist[col].name.title() + ' vs Probability Charge Off', fontsize=20, weight="bold")
        ax.set_xlabel(lc_hist[col].name.title(), fontsize=14)
        ax.set_ylabel('Probability of Charged off', color='b', fontsize=14)
        ax2.set_ylabel('Number of Applicants', color='g', fontsize=14)
        plt.savefig(self.path + 'bii_analysis_' + col + '.jpg')

    def data_description(self, lc_hist):
        # Used descriptive analysis to remove outliers in Income, and understand the
        # data distribution of variables.
        lc_hist.describe().T.to_csv(self.path + 'descriptive_analysis.csv')
        self.univariate_plot(lc_hist, 'loan_amnt', 0)
        self.univariate_plot(lc_hist, 'annual_inc', 0)
        self.univariate_plot(lc_hist, 'loan_status', 1)
        self.univariate_plot(lc_hist, 'purpose', 1, hue='loan_status')
        self.univariate_plot(lc_hist, 'home_ownership', 1, hue='loan_status')
        # Correlation Matrix
        loan_correlation = lc_hist.corr()
        f, ax = plt.subplots(figsize=(14, 9))
        sns.heatmap(loan_correlation,
                    xticklabels=loan_correlation.columns.values,
                    yticklabels=loan_correlation.columns.values, annot=True)
        plt.savefig(self.path + 'corr_heatmap.jpg')
        # Bi-variate Plots
        self.bivariate_plot(lc_hist, 'addr_state')
        self.bivariate_plot(lc_hist, 'purpose', stacked=False)
        self.bivariate_plot(lc_hist, 'annual_inc_binned')
        self.bivariate_plot(lc_hist, 'emp_length')
        self.bivariate_plot(lc_hist, 'fico_range_low_binned')

if __name__ == "__main__":
    sdpv = StartDPV()
    df = sdpv.read_files()
    df = sdpv.data_preprocessing(df)
    sdpv.data_description(df)
    print(df.dtypes)
    # Create Dummy vars
    df = pd.get_dummies(df, columns=['loan_amnt_binned', 'annual_inc_binned','addr_state',
                                     'fico_range_low_binned', 'purpose', 'home_ownership'], drop_first=True)
    drop_features = [ 'fico_range_low', 'percent_bc_gt_75', 'revol_util', 'annual_inc','issue_d', 'earliest_cr_line']
    df = df[df.columns[~df.columns.isin(drop_features)]]
    # sml = StartModelling(df)
    # sml.GBMClassifier()
    # sml.ExtraTreesClassify()


