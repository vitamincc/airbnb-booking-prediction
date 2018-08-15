#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 01:42:51 2018

@author: xucc
"""

def single_model_stacking(clf):
    skf = list(StratifiedKFold(y, 10))
    dataset_blend_train = np.zeros((Xtrain.shape[0],len(set(y.tolist()))))
    # dataset_blend_test = np.zeros((Xtest.shape[0],len(set(y.tolist()))))
    dataset_blend_test_list=[]
    loglossList=[]
    for i, (train, test) in enumerate(skf):
    #     dataset_blend_test_j = []
        X_train = Xtrain[train]
        y_train =dummy_y[train]
        X_val = Xtrain[test]
        y_val = dummy_y[test]
        if clf=='NN_fit':            
            fold_pred,pred=NN_fit(X_train, y_train,X_val,y_val)
        if clf=='xgb_fit':
            fold_pred,pred=xgb_fit(X_train, y_train,X_val,y_val)
        if clf=='lr_fit':
            fold_pred,pred=lr_fit(X_train, y_train,X_val,y_val)
        print('Fold %d, logloss:%f '%(i,log_loss(y_val,fold_pred)))
        dataset_blend_train[test, :] = fold_pred
        dataset_blend_test_list.append( pred )
        loglossList.append(log_loss(y_val,fold_pred))
    dataset_blend_test = np.mean(dataset_blend_test_list,axis=0)
    print('average log loss is :',np.mean(log_loss(y_val,fold_pred)))
    print ("Blending.")
    clf = LogisticRegression(multi_class='multinomial',solver='lbfgs')
    clf.fit(dataset_blend_train, np.argmax(dummy_y,axis=1))
    pred = clf.predict_proba(dataset_blend_test)
    return pred
