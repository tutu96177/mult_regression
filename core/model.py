# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:
# Description:  
# Author:       shichao
# Date:         2019/7/15
#-------------------------------------------------------------------------------

import os, sys
import numpy as np
import pandas as pd
import time

sys.path.append('./')

# 核心代码，设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# 把 y 进行标准化处理: 减去均值除以标准差
def stard_data(df, goals):
    df_target = df[goals]
    for goal in goals:
        df[goal] = df[goal].apply(lambda x: (x-np.mean(df_target[:len(df_target)-12]))/np.std(df_target[:len(df_target)-12]))
    print ()


# 把 y 进行标准化处理: 开平方
def stard_data_aqrt(df, goals):
    for goal in goals:
        df[goal] = np.sqrt(df[goal])
    print()

# 将结果反标准化
def inverse_stard(df, goals, df_res):
    df_target = df[goals]
    # (x-np.mean(df_target[:len(df_target)-12]))/np.std(df_target[:len(df_target)-12])
    for goal in goals:
        goal = int(goal)
        df_res[goal] = df_res[goal].apply(lambda x: round((x*np.std(df_target[:len(df_target)-12]))+np.mean(df_target[:len(df_target)-12]), 2))
    return df_res

# 将结果进行反标准化转化：反平方
def inverse_sqrt(df_res, goals):
    for goal in goals:
        goal = int(goal)
        df_res[goal] = df_res[goal].apply(lambda x: round(x*x, 2))
    return df_res

# 设置交叉验证集的折数
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=10, random_state=42, shuffle=False)
# kf = KFold(n_splits=5, random_state=42, shuffle=False)
# 时间序列分割
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(max_train_size=None, n_splits=13)
def cv_rmse(model, train_X, train_y):
    rmse= np.sqrt(-cross_val_score(model, train_X, train_y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
def cv_mae(model, train_X, train_y):
    cv_mae = np.mean(-cross_val_score(model, train_X, train_y, scoring="neg_mean_absolute_error", cv = kf))
    return cv_mae
def cv_mae_(model, train_X, train_y):
    val_loss = -cross_val_score(model, train_X, train_y, scoring="neg_mean_absolute_error", cv=tscv)
    print('val loss is: {0}'.format(val_loss))
    import matplotlib.pyplot as plt
    plt.plot(val_loss, marker = 'o')
    plt.show()
    cv_mae = np.mean(-cross_val_score(model, train_X, train_y, scoring="neg_mean_absolute_error", cv = tscv))
    return cv_mae

# 选择模型
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
def build_model(train_X, train_y):
    mult_lasso = MultiTaskLassoCV()
    score_lasso = cross_val_score(mult_lasso, train_X, train_y, scoring='neg_mean_absolute_error', cv=kf)
    score_lasso_mean = -score_lasso.mean()
    mult_lasso = mult_lasso.fit(train_X, train_y)

    elncv = ElasticNetCV()
    mult_elncv = MultiOutputRegressor(elncv)
    score_elncv_mean = cv_mae(mult_elncv, train_X, train_y)
    mult_elncv = mult_elncv.fit(train_X, train_y)

    # from sklearn.cross_decomposition import PLSRegression
    # plsr = PLSRegression()
    # mult_plsr = MultiOutputRegressor(plsr)
    # score_plsr = cross_val_score(mult_plsr, train_X, train_y, scoring='neg_mean_absolute_error', cv=kf)
    # score_plsr_mean = -score_plsr.mean()
    # mult_plsr = mult_plsr.fit(train_X, train_y)

    from sklearn.linear_model import BayesianRidge
    b_ridge = BayesianRidge()
    mult_b_ridge = MultiOutputRegressor(b_ridge)
    score_b_ridge = cross_val_score(mult_b_ridge, train_X, train_y, scoring='neg_mean_absolute_error', cv=kf)
    score_b_ridge_mean = -score_b_ridge.mean()
    mult_b_ridge = mult_b_ridge.fit(train_X, train_y)

    import lightgbm as lgb
    lgb = lgb.LGBMRegressor()
    mult_lgb = MultiOutputRegressor(lgb)
    score_lgb = cv_mae(mult_lgb, train_X, train_y)
    mult_lgb = mult_lgb.fit(train_X, train_y)

    import catboost as cab
    cab = cab.CatBoostRegressor()
    mult_cab = MultiOutputRegressor(cab).fit(train_X, train_y)

    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor()
    mult_knn = MultiOutputRegressor(knn)
    score_knn = cv_mae(mult_knn, train_X, train_y)
    mult_knn = mult_knn.fit(train_X, train_y)

    from sklearn.svm import SVR
    #svr = SVR(kernel='rbf')
    svr = SVR()
    mult_svr = MultiOutputRegressor(svr)
    score_svr = cv_mae(mult_svr, train_X, train_y)
    mult_svr = mult_svr.fit(train_X, train_y)

    # 决策树
    dtr = DecisionTreeRegressor()
    mult_dtr = MultiOutputRegressor(dtr)
    # cv_mae_dtr = cv_mae_(mult_dtr, train_X, train_y)
    cv_mae_dtr = cv_mae(mult_dtr, train_X, train_y)
    mult_dtr = mult_dtr.fit(train_X, train_y)
    print()

    # 随机森林
    rf = RandomForestRegressor()
    mult_rf = MultiOutputRegressor(rf)
    # cv_mae_rf = cv_mae_(mult_rf, train_X, train_y)
    cv_mae_rf = cv_mae(mult_rf, train_X, train_y)
    mult_rf = mult_rf.fit(train_X, train_y)

    # adaboost
    adbt = AdaBoostRegressor(random_state=42)
    mult_adbt = MultiOutputRegressor(adbt)
    # cv_mae_adbt = cv_mae_(mult_adbt, train_X, train_y)
    cv_mae_adbt = cv_mae(mult_adbt, train_X, train_y)
    mult_adbt = mult_adbt.fit(train_X, train_y)
    # gbdt

    gbdt = GradientBoostingRegressor()
    mult_gbdt = MultiOutputRegressor(gbdt)
    # cv_mae_gbdt = cv_mae_(mult_gbdt, train_X, train_y)
    cv_mae_gbdt = cv_mae(mult_gbdt, train_X, train_y)
    mult_gbdt = mult_gbdt.fit(train_X, train_y)

    # xgb
    xgb = XGBRegressor()
    mult_xgb = MultiOutputRegressor(xgb)
    #cv_mae_xgb = cv_mae_(mult_xgb, train_X, train_y)
    cv_mae_xgb = cv_mae(mult_xgb, train_X, train_y)
    mult_xgb = mult_xgb.fit(train_X, train_y)

    # 多层感知器, 神经网络
    from sklearn.neural_network import MLPRegressor
    # (8, 10) mae: 973
    # (10, 10) mae: 644
    # (16, 10) mae: 948
    # (22, 10) mae: 647
    # (23, 10) mae: 639
    # (28, 10) mae: 655
    # (23, 8, 10) mae: 804
    # (23, 10, 10) mae: 1028
    # (16, 8, 10) mae: 880
    # (33, 10, 10) mae: 2216
    # (26, 14, 10) mae: 649
    # (26, 12, 10) mae: 679
    # (26, 10, 10) mae: 743
    # (26, 8, 10) mae: 734
    # (26, 6, 10) mae: 632
    # (26, 4, 10) mae: 831
    init_params = {
        'hidden_layer_sizes': (26, 6, 10),
        'solver': 'lbfgs',
        'max_iter': 20000,
        'early_stopping': False,
        'shuffle': False,
        'verbose': True,
        'random_state': 1,
                   }
    mult_mlp = MLPRegressor(**init_params)
    cv_mae_mlp = cv_mae_(mult_mlp, train_X, train_y)
    mult_mlp = mult_mlp.fit(train_X, train_y)
    return mult_adbt

# 保存模型
from sklearn.externals import joblib
def save_model(mult_model, model_save_path):
    """
    :param mult_model: 待保存的模型对象
    :param model_save_path: 保存路径，例如 "./model.pkl"
    :return: 没有返回值
    """
    joblib.dump(mult_model, model_save_path)
# 加载模型
def load_model(model_path):
    """
    :param model_path: 模型路径
    :return: 返回加载后的模型对象
    """
    return joblib.load(model_path)

# 训练模型
def train_model(train_X, train_y):
    boost_initialize_params = {'n_estimators': 295, 'learning_rate': 0.61, 'subsample': 1.0, 'loss': 'quantile'}
    base_initialize_params = {'max_depth': 6, 'min_samples_split': 0.51, 'alpha': 0.11, 'random_state': 42}
    best_params = {}
    best_params.update(boost_initialize_params)
    best_params.update(base_initialize_params)
    model = GradientBoostingRegressor(**best_params)
    mult_model = MultiOutputRegressor(model)
    # 训练多任务
    mult_model = mult_model.fit(train_X, train_y)
    return mult_model



# from parameter_optimize import parameter_optimize
# if __name__ == '__main__':
def run_model():
    # 读取数据
    file_name = 'data' + '.csv'  # 对第 1 阶段数据建模
    file_dir = 'data/'
    file_path = os.path.join(file_dir, file_name)
    df_file = pd.read_csv(file_path, delimiter=',', names=['y1', 'y2', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'])

    # 划分训练集和测试集
    features_list = ['x' + str(x) for x in range(1, 7)]
    target_list = ['y' + str(y) for y in range(1, 3)]
    num_test = int(len(df_file)*0.1)  # 测试集数目
    test_y = df_file.loc[len(df_file) - num_test:][target_list]
    test_y = test_y.reset_index(drop=True)
    df_train = df_file.loc[:len(df_file) - num_test-1]
    df_train = df_train.reset_index(drop=True)
    train_X = df_train[features_list]
    train_y = df_train[target_list]
    df_test = df_file.loc[len(df_file) - num_test:]
    df_test = df_test.reset_index(drop=True)
    test_X = df_test[features_list]

    # 对数据标准化
    # from sklearn import preprocessing
    # scaler = preprocessing.StandardScaler().fit(train_X)
    # train_X = scaler.transform(train_X)
    # test_X = scaler.transform(test_X)

    # 构建模型：选择模型
    mult_model = build_model(train_X, train_y)
    # 模型参数优化，选择最佳参数： adbt
    # parameter_optimize(train_X, train_y)
    # 训练模型:
    mult_model = train_model(train_X, train_y)
    # 测试集预测
    res_pre = mult_model.predict(test_X)
    res_pre = np.round(res_pre)
    df_res = pd.DataFrame(res_pre)
    # 预测值与真实值做差
    diff_y_yhat = df_res.values - test_y
    eval_model = np.mean(np.mean(abs(df_res.values - test_y.values)))
    eval_model_3 = np.mean(abs(df_res.values - test_y.values), axis=1)
    # 测试集评估模型
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    eval_model_1 = mean_absolute_error(test_y.values, df_res, multioutput='raw_values')
    eval_model_2 = mean_absolute_error(test_y.values, df_res)
    rmse = mean_squared_error(test_y.values, df_res)
    print()
