from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import sklearn
import os
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler


class StartModelling:
    def __init__(self, df, val):
        self.train = df
        self.val = val
        self.filePath = os.getcwd()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.train[self.train != 'cc_cons'],
                                                                                self.train.cc_cons, test_size=0.25,
                                                                                random_state=42)

    def rfmodel(self):
        # # Number of trees in random forest
        # n_estimators = [int(x) for x in np.linspace(start=20, stop=200, num=10)]
        # # Number of features to consider at every split
        # max_features = ['auto', 'sqrt']
        # # Maximum number of levels in tree
        # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        # max_depth.append(None)
        # # Minimum number of samples required to split a node
        # min_samples_split = [2, 5, 10]
        # # Minimum number of samples required at each leaf node
        # min_samples_leaf = [1, 2, 4]
        # # Method of selecting samples for training each tree
        # bootstrap = [True, False]
        # random_grid = {'n_estimators': n_estimators,
        #                'max_features': max_features,
        #                'max_depth': max_depth,
        #                'min_samples_split': min_samples_split,
        #                'min_samples_leaf': min_samples_leaf,
        #                'bootstrap': bootstrap}
        # print(random_grid)
        # rf = RandomForestRegressor()
        # rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
        #                                random_state=42, n_jobs=-1)
        # rf_random.fit(self.X_train, self.y_train)
        # print(rf_random.best_params_)
        # gsc = GridSearchCV(
        #     estimator=RandomForestRegressor(),
        #     param_grid={
        #         'max_depth': range(3, 10),
        #         'n_estimators': (10, 50),
        #     },
        #     cv=5, scoring='neg_mean_squared_log_error', verbose=2, n_jobs=-1)
        # grid_result = gsc.fit(self.X_train, self.y_train)
        # best_params = grid_result.best_params_
        # print(best_params)
        rfr = RandomForestRegressor(max_depth=15,
                                    n_estimators=1200,
                                    min_samples_split=5,
                                    min_samples_leaf=5,
                                    max_features=None,
                                    oob_score=True,
                                    random_state=42)
        kf = KFold(n_splits=12, random_state=42, shuffle=True)
        scores = cross_val_score(rfr, self.X_train, self.y_train, cv=kf, scoring='neg_mean_squared_log_error')
        print(np.mean(scores), np.sqrt(scores))
        # predictionsTest = cross_val_predict(rfr, self.X_test, self.y_test, cv=10)
        # print(predictionsTest)
        # print(sklearn.metrics.mean_squared_log_error(self.y_test, predictionsTest))
        rfr.fit(self.X_train, self.y_train)
        predictions = rfr.predict(self.X_test)
        print(sklearn.metrics.mean_squared_log_error(self.y_test, predictions)*100)
        errors = abs(predictions - self.y_test)
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
        mape = 100 * (errors / self.y_test)
        accuracy = 100 - np.mean(mape)
        print('Accuracy:', round(accuracy, 2), '%.')
        # predval = rfr.predict(self.val)
        # return predictions, predval

    def svrmodel(self):
        svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))
        svr.fit(self.X_train, self.y_train)
        predictions = svr.predict(self.X_test)
        print(svr.score(self.X_test, self.y_test))
        print(sklearn.metrics.mean_squared_log_error(self.y_test, predictions))
        # predvalsvr = svr.predict(self.val)
        # return predvalsvr
