# coding=utf-8
"""build multi model to train the data

"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import time
import xgboost as xgb




def RfModelDecision(train_x, label, Ptest_x, Ntest_x):

    print 'training random forest model...'
    rf_model= RandomForestClassifier(n_estimators = 60)
    rf_model.fit(train_x, label)

    Ppred_rf = rf_model.predict_proba(Ptest_x)
    Npred_rf = rf_model.predict_proba(Ntest_x)

    test_label = [1]*len(Ppred_rf) + [0]*len(Npred_rf)
    score = Ppred_rf.tolist() + Npred_rf.tolist()
    score_rf = [x[1] for x in score]
    auc = roc_auc_score(test_label,score_rf)
    print 'rf_AUC:', auc

    return score_rf, rf_model

def SVMModelDecision(train_x, label, Ptest_x, Ntest_x):

    print 'training svm model...'
    svm_model= svm.SVC(C=35, gamma=0.005, kernel='rbf', probability=True)
    svm_model.fit(train_x, label)

    Ppred_svm = svm_model.predict_proba(Ptest_x)
    Npred_svm = svm_model.predict_proba(Ntest_x)

    test_label = [1]*len(Ppred_svm) + [0]*len(Npred_svm)
    score = Ppred_svm.tolist() + Npred_svm.tolist()
    score_svm = [x[1] for x in score]
    auc = roc_auc_score(test_label,score_svm)
    print 'svm_AUC:', auc

    return score_svm, svm_model


def run_xgb(train, test, features, target, random_state=0):
    start_time = time.time()
    objective = "reg:linear"
    booster = "gbtree"
    eval_metric = "rmse"
    eta = 0.1
    max_depth = 3
    subsample = 0.7
    colsample_bytree = 0.7
    silent = 1

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": objective,
        #         "num_class": 2,
        "booster" : booster,
        "eval_metric": eval_metric,
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": silent,
        "seed": random_state,
    }
    num_boost_round = 200
    early_stopping_rounds = 20
    test_size = 0.2

    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    y_train = X_train[target]
    y_valid = X_valid[target]
    dtrain = xgb.DMatrix(X_train[features], y_train)
    dvalid = xgb.DMatrix(X_valid[features], y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration)

    score = mean_squared_error(y_valid.tolist(), check)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features]), ntree_limit=gbm.best_iteration)

    training_time = round((time.time() - start_time)/60, 2)
    print('Training time: {} minutes'.format(training_time))

    print(gbm)

    # To save logs
    explog = {}
    explog['features'] = features
    explog['target'] = target
    explog['params'] = {}
    explog['params']['objective'] = objective
    explog['params']['booster'] = booster
    explog['params']['eval_metric'] = eval_metric
    explog['params']['eta'] = eta
    explog['params']['max_depth'] = max_depth
    explog['params']['subsample'] = subsample
    explog['params']['colsample_bytree'] = colsample_bytree
    explog['params']['silent'] = silent
    explog['params']['seed'] = random_state
    explog['params']['num_boost_round'] = num_boost_round
    explog['params']['early_stopping_rounds'] = early_stopping_rounds
    explog['params']['test_size'] = test_size
    explog['length_train']= len(X_train.index)
    explog['length_valid']= len(X_valid.index)
    # explog['gbm_best_iteration']=
    explog['score'] = score
    explog['training_time'] = training_time




    return test_prediction.tolist(), score, explog

def ROC_Plot(predict_vec, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, predict_vec)
    roc_auc = auc(fpr, tpr)
    print 'AUC:', roc_auc
    #plot the false positive and true positive
    plt.plot(fpr, tpr, lw=1)