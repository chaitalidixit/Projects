# Load System Libraries
import pandas as pd
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from foreclosureprediction.ForeclosureModelling import ForeclosureModelling

'''
Dataprocessing class: reads the files, merges the files, performs feature engineering and creates an
output which can be used for data exploration/imputation.
'''


class ForeclosureDataProcessing:

    def __init__(self):
        self.filePath = os.getcwd() + '/foreclosureprediction/data/'
        self.dataDict = {}

    # Read the files and store in a dictionary
    def readFiles(self):
        self.dataDict['LoanFinSheet'] = pd.read_csv(self.filePath + 'LoanFinSheet.csv', infer_datetime_format=True,
                                                    thousands=',', parse_dates=['INTEREST_START_DATE',
                                                                                'AUTHORIZATIONDATE',
                                                                                'LAST_RECEIPT_DATE'],
                                                    dtype={'NPA_IN_LAST_MONTH': str, 'NPA_IN_CURRENT_MONTH': str})
        self.dataDict['CustomerDemo'] = pd.read_csv(self.filePath + 'CustomerDemo.csv', dtype={'BRANCH_PINCODE': str},
                                                    na_values=' ')
        self.dataDict['CustomerMails'] = pd.read_csv(self.filePath + 'CustomerMails.csv', encoding="ISO-8859-1",
                                                     infer_datetime_format=True, parse_dates=['Date'])
        self.dataDict['train_foreclosure'] = pd.read_csv(self.filePath + 'train_foreclosure.csv')
        self.dataDict['BaseRateHistorical'] = pd.read_csv(self.filePath + 'BaseRateHistorical.csv',
                                                          infer_datetime_format=True, parse_dates=['Effective Date'],
                                                          index_col=['Effective Date'])

    # Adding SBI base rates for past 10 years
    def addBaseRate(self, row):
        try:
            row['BASE_RATE@INCEPTION'] = self.dataDict['BaseRateHistorical'].iloc[self.dataDict['BaseRateHistorical']. \
                index.get_loc(row['AUTHORIZATIONDATE'], method='bfill')][0]
            row['BASE_RATE@EMIRECEIPTDATE'] = \
                self.dataDict['BaseRateHistorical'].iloc[self.dataDict['BaseRateHistorical']. \
                    index.get_loc(row['LAST_RECEIPT_DATE'], method='bfill')][0]
        except Exception:
            row['BASE_RATE@INCEPTION'] = 0
            row['BASE_RATE@EMIRECEIPTDATE'] = 0
        return row

    # Processing and Merging datasets to create a final dataframe with unique records
    def dataprocessor(self):
        dfdemoLoan = pd.merge(self.dataDict['LoanFinSheet'], self.dataDict['CustomerDemo'], how='left',
                              on='CUSTOMERID')
        dictMail = self.analyseMails()
        dfdailymailcnt = pd.merge(dfdemoLoan, dictMail['freqMailer>5'], how='left', on='CUSTOMERID')
        dfdailymailcnt = dfdailymailcnt.apply(self.addBaseRate, axis=1)
        dfdailymailcnt.to_csv(self.filePath + 'mailsbaserate.csv', index=False)
        # adding 2 new features here
        dfdailymailcnt['PAID_PRINCIPALpct'] = dfdailymailcnt.groupby(['AGREEMENTID'])['PAID_PRINCIPAL'].transform(
            lambda x: x.pct_change())
        dfdailymailcnt.PAID_PRINCIPALpct.replace([np.inf, -np.inf], np.nan, inplace=True)
        dfdailymailcnt['PAID_PRINCIPALpct'].fillna(0, inplace=True)
        dfdailymailcnt['TENURECHANGE'] = dfdailymailcnt['CURRENT_TENOR'] - dfdailymailcnt['ORIGNAL_TENOR']
        dfdailymailcnt['loanprepay'] = np.where(np.logical_or(dfdailymailcnt['TENURECHANGE'] < 0,
                                                              np.logical_and(dfdailymailcnt['PAID_PRINCIPALpct'] > 0.05,
                                                                             np.logical_and(np.greater(
                                                                                 dfdailymailcnt['EXCESS_ADJUSTED_AMT'],
                                                                                 0), \
                                                                                 np.less_equal(
                                                                                     dfdailymailcnt[
                                                                                         'NET_RECEIVABLE'],
                                                                                     0)))), 1, 0)
        dfdailymailcnt['loanrefinance'] = np.where(dfdailymailcnt['PAID_PRINCIPALpct'] < 0, 1, 0)

        dfnewFeatures = self.featureEngg(dfdailymailcnt)
        dfDatawithTarget = pd.merge(dfnewFeatures, self.dataDict['train_foreclosure'], how='left',
                                    on='AGREEMENTID')
        dfDatawithTarget.to_csv(self.filePath + 'allDataNewFeatures.csv', index=False)
        dfgrpd = dfDatawithTarget.groupby(['AGREEMENTID']).mean().reset_index()
        dfUniqueData = (dfDatawithTarget.merge(dfgrpd, on=['AGREEMENTID'], suffixes=('', '_mean'))
                        .drop(list(dfgrpd.columns[1:]), axis=1)
                        .groupby(['AGREEMENTID'])
                        .first()
                        .reset_index()
                        )
        dfUniqueData.columns.str.replace('_mean', '')
        dfUniqueData.columns = dfUniqueData.columns.str.replace('_mean', '')
        dfUniqueData.to_csv(self.filePath + 'dfUnique.csv', index=False)
        return dfDatawithTarget

    # Analysis frequency of customer communications
    def analyseMails(self):
        mailAnalysis = {}
        dfMail = self.dataDict['CustomerMails'].copy()
        dfMail.rename(columns={'Masked_CustomerID': 'CUSTOMERID'}, inplace=True)
        mailAnalysis['freqMailerCust'] = dfMail.CUSTOMERID.value_counts().to_frame(name='custmailfreq').reset_index()
        mailAnalysis['freqMailerCust'].rename(columns={'index': 'CUSTOMERID'}, inplace=True)
        mailAnalysis['freqMailer>5'] = mailAnalysis['freqMailerCust'][
            mailAnalysis['freqMailerCust']['custmailfreq'] > 5]
        mailAnalysis['grpdSubType'] = dfMail.groupby(['CUSTOMERID', 'SubType']).size(). \
            to_frame(name='subtypecount').reset_index()
        with open(self.filePath + 'mailAnalysisdict.txt', 'wb') as output:
            pickle.dump(mailAnalysis, output)
        return mailAnalysis

    # Create new features from the existing data to be used for modelling
    def featureEngg(self, dfgrpd):
        df = dfgrpd.copy()
        df['ROICHANGE'] = (df['CURRENT_ROI'] - df['ORIGNAL_ROI']) / 100
        df['ORIG_BASERATE'] = (df['ORIGNAL_ROI'] - df['BASE_RATE@INCEPTION']) / 100
        df['CURR_BASERATE'] = (df['CURRENT_ROI'] - df['BASE_RATE@EMIRECEIPTDATE']) / 100
        df['GROSSINCOMEIMPUTED'] = df['LOAN_AMT'] / df['FOIR']
        df['ASSESTVALUE'] = df['LOAN_AMT'] / df['NET_LTV']
        df['LOANAGE'] = (df['LAST_RECEIPT_DATE'] - df['INTEREST_START_DATE']).dt.days / 30.0
        df['NPALoan30-60'] = np.where(29 < df['DPD'], 1, 0)
        df['NPALoan60-90'] = np.where(60 < df['DPD'], 1, 0)
        df['NPALoanGT90'] = np.where(df['DPD'] > 90, 1, 0)
        df['contractualpay'] = np.where(df['DPD'] < 30, 1, 0)
        df['volparpay'] = np.where(df['loanprepay'] > 0, 1, 0)
        df['loanrefinance'] = np.where(df['loanrefinance'] > 0, 1, 0)
        df['default'] = np.where(np.logical_or(df['NPALoan30-60'] == 1, np.logical_or(df['NPALoan60-90'] == 1,
                                                                                      df['NPALoanGT90'] == 1)), 1, 0)
        return df


'''
DataExploration: creates heatmap, pairplots(uni/bi variate), bar plots to analyse features
and creates the train/test datasets to be used for modelling.
'''


class DataExploration:
    def __init__(self, uniqueDatadf):
        self.df = uniqueDatadf
        self.filePath = os.getcwd() + '/foreclosureprediction/data/'

    # Created a 3 featured target variable of voluntary prepayment, default and contractual payments customers
    def createNewtarget(self, row):
        if row['volparpay'] == 1:
            return 1
        if row['default'] == 1:
            return 2
        if row['contractualpay'] == 1:
            return 0
        else: return 1

    # Plot to see the Age distribution for contractual and voluntary prepayments
    def binnedAgeplot(self, train):
        dfVis = train.copy()
        dfgrpd = dfVis.groupby(dfVis.AGE).sum()
        dfAgeCntofPays = dfgrpd[['contractualpay', 'volparpay']]
        f = dfAgeCntofPays.plot(kind='bar').get_figure()
        f.show()
        f.savefig(self.filePath + 'agevscntofpays.png')

    # Plot to view the impact of ROI change on payments
    def biVariateplot(self, train):
        dfroi = train.copy()
        bins = [-0.15, -0.1, -0.008, -0.006, -0.004, -0.0002, -0.00001, 0, 0.00001, 0.0002, 0.004, 0.006, 0.008, 0.1,
                0.15]
        dfroi['ROIbinned'] = pd.cut(dfroi['ROICHANGE'], bins)
        dfgrpdroi = dfroi.groupby(dfroi.ROIbinned).sum()
        dfROICntofPays = dfgrpdroi[['default', 'volparpay']]
        dfROICntofPays.to_csv(self.filePath + 'dfROICntofPays.csv')
        g = dfROICntofPays.plot(kind='bar', fontsize=5, rot=45).get_figure()
        g.savefig(self.filePath + 'roivscntofpays.png')

        dfDate = train.copy()
        dfgrpdate = dfDate.groupby(dfroi.LAST_RECEIPT_DATE).sum()
        dfLastDate = dfgrpdate[['default', 'volparpay']]
        dfLastDate.plot(kind='bar', fontsize=5, rot=10, figsize=(50, 20))
        plt.tick_params(axis='both', which='both', labelsize=5)
        plt.xticks(plt.xticks()[0][1::20],
                   plt.xticks()[1][1::20])
        plt.tight_layout()
        plt.savefig(self.filePath + 'lastdate.png')

        dfLoanAge = train.copy()
        dfgrpdla = dfLoanAge.groupby(dfLoanAge.LOANAGE).sum()
        dfLA = dfgrpdla[['default', 'volparpay']]
        dfLA.plot(kind='bar', fontsize=5, rot=10, figsize=(50, 20))
        plt.tick_params(axis='both', which='both', labelsize=5)
        plt.xticks(plt.xticks()[0][1::20],
                   plt.xticks()[1][1::20])
        plt.tight_layout()
        plt.savefig(self.filePath + 'loanage.png')

        dfLoanAmt = train.copy()
        bins = [0, 0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.8]
        dfLoanAmt['LAbinned'] = pd.cut(dfLoanAmt['LOAN_AMT'], bins)
        dfgrpdlm = dfLoanAmt.groupby(dfLoanAmt.LAbinned).sum()
        dfLM = dfgrpdlm[['default', 'volparpay']]
        h = dfLM.plot(kind='bar', fontsize=5, rot=10, figsize=(50, 20)).get_figure()
        h.savefig(self.filePath + 'loanamt.png')

        dfFOIR = train.copy()
        #        bins = [0, 0.0001,0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.8]
        dfFOIR['FOIRbinned'] = pd.cut(dfFOIR['FOIR'], bins)
        dfgrpdfr = dfFOIR.groupby(dfFOIR.FOIR).sum()
        dffr = dfgrpdfr[['default', 'volparpay']]
        j = dffr.plot(kind='bar', fontsize=7, rot=10, figsize=(50, 20)).get_figure()
        j.savefig(self.filePath + 'foir.png')

        dfLTV = train.copy()
        bins = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        dfLTV['LTVbinned'] = pd.cut(dfLTV['NET_LTV'], bins)
        dfgrpdlv = dfLTV.groupby(dfLTV.LTVbinned).sum()
        dflv = dfgrpdlv[['default', 'volparpay']]
        k = dflv.plot(kind='bar', fontsize=7, rot=10, figsize=(50, 20)).get_figure()
        k.savefig(self.filePath + 'ltv.png')

    # General Data descriptions stats
    def dataDescription(self, dfImpute):
        dfdescp = dfImpute.copy()
        # create stats for the dataset
        dfStats = dfdescp.describe().transpose()
        dfStats.to_csv(self.filePath + 'dataStats.csv')
        # check for null values to assist missing value imputation
        total = dfdescp.isnull().sum().sort_values(ascending=False)
        percent = (dfdescp.isnull().sum() / dfdescp.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        # used this for data imputation
        missing_data.to_csv(self.filePath + "missing_data.csv")

    # Pair plots and heatmaps to identify multi-collinearity
    def pairPlots(self, train):
        dfpair = train.copy()
        # create pairplots
        cols = ['FORECLOSURE', 'LOAN_AMT', 'ROICHANGE', 'GROSSINCOMEIMPUTED', 'ASSESTVALUE', 'AGE',
                'contractualpay', 'volparpay', 'ORIG_BASERATE', 'CURR_BASERATE', 'OUTSTANDING_PRINCIPAL',
                'PAID_PRINCIPAL', 'PAID_INTEREST', 'AGEPATTERN', 'NPALoan30-60', 'NPALoan60-90', 'NPALoanGT90',
                'loanrefinance',
                'LOANAGE', 'EMI_DUEAMT', 'EMI_RECEIVED_AMT', 'custmailfreq']

        sns.pairplot(dfpair[cols], size=2.5)
        plt.savefig(self.filePath + "pairplot.png")

        # create heatmap using integer vars
        dfdescpSub = dfpair.select_dtypes(exclude=['object', 'datetime'])
        corr = dfdescpSub.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        f, ax = plt.subplots(figsize=(80, 100))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.savefig(self.filePath + "heatmap.png")

    # Impute data to bring all variables within same numeric domain
    def dataImputation(self):
        dfImpute = self.df.copy()
        # added below after taking a look at the dataprocessor dataset
        dfImpute.GROSSINCOMEIMPUTED = np.where(~np.isfinite(dfImpute.GROSSINCOMEIMPUTED),
                                               dfImpute.EMI_DUEAMT / 0.25, dfImpute.GROSSINCOMEIMPUTED)
        dfImpute.GROSSINCOMEIMPUTED = np.where(dfImpute.GROSSINCOMEIMPUTED < 0, dfImpute.EMI_DUEAMT / 0.25,
                                               dfImpute.GROSSINCOMEIMPUTED)
        dfImpute.GROSSINCOMEIMPUTED = np.where(dfImpute.GROSSINCOMEIMPUTED == 0, (dfImpute.LOAN_AMT / 30) * 12,
                                               dfImpute.GROSSINCOMEIMPUTED)
        # added below post observing the dataprocessor dataset, created some dist plots in excel

        dfImpute['custmailfreq'].fillna(0, inplace=True)
        dfImpute['TENURECHANGE'].fillna(0, inplace=True)
        dfImpute['LAST_RECEIPT_DATE'].fillna(value=dfImpute['INTEREST_START_DATE'], inplace=True)
        dfImpute['NO_OF_DEPENDENT'].fillna(0, inplace=True)
        dfImpute['LOANAGE'] = (dfImpute['LAST_RECEIPT_DATE'] - dfImpute['INTEREST_START_DATE']).dt.days / 30.0
        dfImpute['LOANAGE'].fillna(0, inplace=True)
        le = LabelEncoder()
        dfImpute['CITYEncode'] = le.fit_transform(dfImpute['CITY'].astype(str))
        dfImpute['PRODUCTEncode'] = le.fit_transform(dfImpute['PRODUCT'].astype(str))
        dfImpute['Targetnew'] = dfImpute.apply(self.createNewtarget, axis=1)
        # Created the dataset foreclosure clients
        dfForeclosed = dfImpute[dfImpute.FORECLOSURE.notnull()]

        # Created the hold-out dataset
        val = dfImpute[dfImpute.FORECLOSURE.isnull()]

        dfForeclosed.to_csv(self.filePath + 'dfForeclosed.csv', index=False)
        val.to_csv(self.filePath + 'val_fromcode.csv', index=False)

        X_train, y_train, X_test, y_test, X_val, y_val = self.splitDataNVisualise(dfForeclosed, val)

        return X_train, y_train, X_test, y_test, X_val, y_val

    def splitDataNVisualise(self, dfForeclosed, val):
        print('Begin Data Split')
        dfForeclosed.rename(columns={'NPALoan>90': 'NPALoanGT90'}, inplace=True)
        val.rename(columns={'NPALoan>90': 'NPALoanGT90'}, inplace=True)
        dfdict = dict(tuple(dfForeclosed.groupby('AGREEMENTID')))
        dfTrain = pd.DataFrame(columns=dfForeclosed.columns)
        dfTest = pd.DataFrame(columns=dfForeclosed.columns)
        for i in list(dfdict.keys())[:12000]:
            dfTrain = dfTrain.append(dfdict[i], ignore_index=True)
            print(len(dfTrain.index))
        dfTrain.reset_index(inplace=True, drop=True)

        dfTrain['AGE'].fillna(value=dfTrain['AGE'].mean(), inplace=True)
        dfTrain['AGEPATTERN'] = abs(30 - abs(30 - dfTrain.AGE))
        dfTrain.to_csv(self.filePath + 'dfTrain.csv', index=False)
        #        self.pairPlots(dfTrain)
        #        self.binnedAgeplot(dfTrain)
        #        self.biVariateplot(dfTrain)
        #        self.dataDescription(dfTrain)

        X_train = dfTrain.loc[:, dfTrain.columns != 'FORECLOSURE']
        X_train.reset_index(inplace=True, drop=True)
        y_train = dfTrain['FORECLOSURE']
        y_train.reset_index(inplace=True, drop=True)

        for i in list(dfdict.keys())[12000:]:
            dfTest = dfTest.append(dfdict[i], ignore_index=True)
        dfTest.reset_index(inplace=True, drop=True)
        X_test = dfTest.loc[:, dfTest.columns != 'FORECLOSURE']
        X_test.reset_index(inplace=True, drop=True)
        y_test = dfTest['FORECLOSURE']
        y_test.reset_index(inplace=True, drop=True)

        X_val = val.loc[:, val.columns != 'FORECLOSURE']
        X_val.reset_index(inplace=True, drop=True)
        y_val = val['FORECLOSURE']
        y_val.reset_index(inplace=True, drop=True)

        scaler = MinMaxScaler()
        X_train[['EMI_DUEAMT', 'EMI_RECEIVED_AMT', 'LOAN_AMT', 'GROSSINCOMEIMPUTED', 'ASSESTVALUE', 'LOANAGE', \
                 'OUTSTANDING_PRINCIPAL', 'PAID_PRINCIPAL', 'PAID_INTEREST', 'MONTHOPENING', 'TENURECHANGE']] = \
            scaler.fit_transform(
                X_train[['EMI_DUEAMT', 'EMI_RECEIVED_AMT', 'LOAN_AMT', 'GROSSINCOMEIMPUTED', 'ASSESTVALUE', \
                         'LOANAGE', 'OUTSTANDING_PRINCIPAL', 'PAID_PRINCIPAL', 'PAID_INTEREST', 'MONTHOPENING', \
                         'TENURECHANGE']])

        X_test[['EMI_DUEAMT', 'EMI_RECEIVED_AMT', 'LOAN_AMT', 'GROSSINCOMEIMPUTED', 'ASSESTVALUE', 'LOANAGE', \
                'OUTSTANDING_PRINCIPAL', 'PAID_PRINCIPAL', 'PAID_INTEREST', 'MONTHOPENING', 'TENURECHANGE']] = \
            scaler.transform(X_test[['EMI_DUEAMT', 'EMI_RECEIVED_AMT', 'LOAN_AMT', 'GROSSINCOMEIMPUTED', 'ASSESTVALUE', \
                                     'LOANAGE', 'OUTSTANDING_PRINCIPAL', 'PAID_PRINCIPAL', 'PAID_INTEREST',
                                     'MONTHOPENING', \
                                     'TENURECHANGE']])
        X_val[['EMI_DUEAMT', 'EMI_RECEIVED_AMT', 'LOAN_AMT', 'GROSSINCOMEIMPUTED', 'ASSESTVALUE', 'LOANAGE', \
               'OUTSTANDING_PRINCIPAL', 'PAID_PRINCIPAL', 'PAID_INTEREST', 'MONTHOPENING', 'TENURECHANGE']] = \
            scaler.transform(X_val[['EMI_DUEAMT', 'EMI_RECEIVED_AMT', 'LOAN_AMT', 'GROSSINCOMEIMPUTED', 'ASSESTVALUE', \
                                    'LOANAGE', 'OUTSTANDING_PRINCIPAL', 'PAID_PRINCIPAL', 'PAID_INTEREST',
                                    'MONTHOPENING', \
                                    'TENURECHANGE']])
        print('End Data Split')

        X_test['AGE'].fillna(value=X_train['AGE'].mean(), inplace=True)
        X_test['AGEPATTERN'] = abs(30 - abs(30 - X_test.AGE))

        X_val['AGE'].fillna(value=X_train['AGE'].mean(), inplace=True)
        X_val['AGEPATTERN'] = abs(30 - abs(30 - X_val.AGE))

        X_train.to_csv(self.filePath + 'X_train.csv', index=False)
        y_train.to_csv(self.filePath + 'y_train.csv', index=False)
        X_test.to_csv(self.filePath + 'X_test.csv', index=False)
        y_test.to_csv(self.filePath + 'y_test.csv', index=False)
        X_val.to_csv(self.filePath + 'X_val.csv', index=False)
        y_val.to_csv(self.filePath + 'y_val.csv', index=False)

        return X_train, y_train, X_test, y_test, X_val, y_val


# Call the functions from within this module
if __name__ != '__main__':
    fdp = ForeclosureDataProcessing()
    fdp.readFiles()
    uniqueDatadf = fdp.dataprocessor()
    # uniqueDatadf = pd.read_csv(os.getcwd() + '/foreclosureprediction/data/allDataNewFeatures.csv', infer_datetime_format=True,
    #                            parse_dates=['INTEREST_START_DATE', 'AUTHORIZATIONDATE', 'LAST_RECEIPT_DATE'],
    #                            dtype={'NPALoan30-60': np.float64, 'NPALoan60-90': np.float64, 'NPALoan>90': np.float64,
    #                                   'contractualpay': np.float64, 'volparpay': np.float64, 'default': np.float64,
    #                                   'ORIGNAL_TENOR': np.float64})
    # dex = DataExploration(uniqueDatadf)
    # X_train, y_train, X_test, y_test, X_val, y_val = dex.dataImputation()
    X_train = pd.read_csv(os.getcwd() + '/foreclosureprediction/data/X_train.csv',infer_datetime_format=True,
                               parse_dates=['INTEREST_START_DATE','AUTHORIZATIONDATE', 'LAST_RECEIPT_DATE'])
    X_test = pd.read_csv(os.getcwd() + '/foreclosureprediction/data/X_test.csv',infer_datetime_format=True,
                              parse_dates=['INTEREST_START_DATE','AUTHORIZATIONDATE', 'LAST_RECEIPT_DATE'])
    X_val = pd.read_csv(os.getcwd() + '/foreclosureprediction/data/X_val.csv',infer_datetime_format=True,
                              parse_dates=['INTEREST_START_DATE','AUTHORIZATIONDATE', 'LAST_RECEIPT_DATE'])
    y_train = pd.read_csv(os.getcwd() + '/foreclosureprediction/data/y_train.csv', header=None)
    y_test = pd.read_csv(os.getcwd() + '/foreclosureprediction/data/y_test.csv', header=None)
    y_val = pd.read_csv(os.getcwd() + '/foreclosureprediction/data/y_val.csv', header=None)
    fm = ForeclosureModelling(X_train, y_train, X_test, y_test, X_val, y_val)
    # fm.featureSelection()
    # fm.logisticRegressionsk()
    # fm.boostedTrees()
    fm.adaBoost()
    # fm.logisticRegressiontf()
#     fm.BoostedTreesClassifiertf()


#        dfImpute['contractualpay'] = np.where(dfImpute['contractualpay'] > 0, 1, 0)
#        dfImpute['volparpay'] = np.where(dfImpute['volparpay'] > 0, 1, 0)
#        dfImpute['loanrefinance'] = np.where(dfImpute['loanrefinance'] > 0, 1, 0)
#        dfImpute['default'] = np.where(dfImpute['default'] > 0, 1, 0)
