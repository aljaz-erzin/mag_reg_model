import numpy as np
import decimal
from common import recreate_input_dataset, standardization_train, standardization_other, one_hot_encoding, \
    convert_string, impute_missing_values, correlation, build_model, predict, rmse, r2, feature_target_correlation, \
    find_worst_feature, eliminate_constants, define_weights, throw_features_away
from db_read_write import PES_select_active_ids, FEA_insert_data, TRMS_select_active_by_pesid, MLA_select_by_id, \
    FIM_insert_data, PRE_insert_multitarget_data, TRMS_update_eval_scores, UNDS_select_active_ids, TRMS_select_by_id, \
    PES_select_by_id, select_data, TDS_update_weights, DIS_get_column, RPEDPLOT_get_trms_id, \
    PRE_insert_single_target_data
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import linear_model
from plots import plot_distribution, plot_predictions_neto
import sys
from db_connection import output_file
from datetime import datetime
import pandas as pd

def main():

    train_plot = False

    """
        PLOTS
    """
    if train_plot == True:
        left_init_cut, right_init_cut = -2, +2
        left_cut = True
        right_cut = True
        factor_1 = 2
        factor_2 = 2
        table = 'INSERT_TABLE'
        column = 'RATIO'
        test_train_type = "TRAIN"  # TEST TRAIN

        """
        df = DIS_get_column(table, column)
        data = df.values[:, 0].astype(float)
        tuple_id_w, weights = define_weights(data, data, left_init_cut, right_init_cut, left_cut, right_cut, factor_1, factor_2)"""

        # plot_predictions_neto(table, 'PRED_NETO_RATIO', 'RATIO', test_train_type, TRMS_ID)
        datefrom = '1900-01-01'
        plot_distribution(table, column, left_init_cut, right_init_cut, left_cut, right_cut, factor_1, factor_2,
                          datefrom)

        # plot_chebisev(table, 'NETO/BRUTO RATIO', left_init_cut, right_init_cut, left_cut, right_cut, 3, 2)
        exit()

    """
        1.) MODEL TRAIN AND TEST
    """

    PES_Active_ids = PES_select_active_ids()
    for _, PES in PES_Active_ids.iterrows():
        """if PES['PES_ID'] <= 255:
            continue"""
        print('----------------------------------- Execution start: PES_ID = %d -----------------------------------' %
              PES['PES_ID'])
        """
            DATA LOAD, TRANSFORM
        """
        # Init
        PES_OneHotEncodeFeatures = convert_string(PES['PES_OneHotEncodeFeatures'])
        PES_TargetVariables = convert_string(PES['PES_TargetVariables'])
        PES_ExcludedFeatures = convert_string(PES['PES_ExcludedFeatures'])
        PES_ExcludedFeatures.append('ID')

        # Load data
        left = PES['PES_left_border']
        if left is None or np.isnan(left):
            left = -2
        right = PES['PES_right_border']
        if right is None or np.isnan(right):
            right = 2
        left_odlocba = PES['PES_Odlocba_left_border']
        if left_odlocba is None or np.isnan(left_odlocba):
            left_odlocba = -2
        right_odlocba = PES['PES_Odlocba_right_border']
        if right_odlocba is None or np.isnan(right_odlocba):
            right_odlocba = 2
        left_kolicina = PES['PES_Kolicina_left']
        if left_kolicina is None or np.isnan(left_kolicina) :
            left_kolicina = -99999999
        right_kolicina = PES['PES_Kolicina_right']
        if right_kolicina is None or np.isnan(right_kolicina):
            right_kolicina = 99999999
        left_kubatura = PES['PES_Kubature_left']
        if left_kubatura is None or np.isnan(left_kubatura) :
            left_kubatura = -99999999
        right_kubatura = PES['PES_Kubature_right']
        if right_kubatura is None or np.isnan(right_kubatura):
            right_kubatura = 99999999
        df = select_data(PES, left, right, left_odlocba, right_odlocba, left_kolicina, right_kolicina, left_kubatura, right_kubatura, PES_TargetVariables)

        print("NUMBER OF SAMPLES:", df.values.shape[0])

        # Store continous features:
        continous_features = np.array(convert_string(PES['PES_ContinousFeatures']))
        # print(continous_features)

        # convert selected columns to float data type
        # decimal_cols = df.select_dtypes(include=['decimal']).columns

        # Handle categorical features
        if PES_OneHotEncodeFeatures is not None:
            df = one_hot_encoding(df, PES_OneHotEncodeFeatures)

        # Prepare features/target split
        X = df.drop(PES_TargetVariables, axis=1)
        X_ForLearning = np.array([not (i in PES_ExcludedFeatures) for i in X.columns])
        y = df[['ID'] + PES_TargetVariables]
        y_ForLearning = np.array([not (i in PES_ExcludedFeatures) for i in y.columns])

        # Eliminate constant features (threshold=0) and quazy constant (threshold=0.01)
        # constant_columns = eliminate_constants(X.loc[:, X_ForLearning], 0)
        constant_columns = eliminate_constants(X.loc[:, X_ForLearning], PES['PES_exclude_constants_threshold'])  # 0.001
        print('NUMBER OF (NEAR) CONSTANT COLUMNS = ' + str(len(constant_columns)) + '. CONSTANT COLUMNS (DROPED): ',
              constant_columns)
        X_ForLearning[X.columns.get_indexer(constant_columns)] = False

        # Impute missing feature values
        X = impute_missing_values(X, [col for col in X.columns if col not in PES_ExcludedFeatures], strategy='mean')

        # Apply PCA for Kubature
        pca_cols = convert_string(PES['PES_PCA_Features'])
        pca_cols = np.array(pca_cols) if pca_cols is not None else None
        if pca_cols is not None and pca_cols.shape[0] > 0:
            pca = PCA(n_components=PES['PES_PCA_variance_kept'])
            #indices = [X.columns.get_loc(col) for col in pca_cols]
            if X.shape[0] > 0:
                X_pca = pca.fit_transform(X[pca_cols])
                pca_new_cols = ['PCA_Kubatura_'+str(i) for i in range(X_pca.shape[1])]
                pca_new_cols = ['PCA_Kubatura_'+str(i) for i in range(X_pca.shape[1])]
                column_indices = [X.columns.get_loc(col) for col in pca_cols]
                X_ForLearning = np.delete(X_ForLearning, column_indices)
                X_ForLearning = np.append(X_ForLearning, [True for _ in range(X_pca.shape[1])])
                X = X.drop(columns=pca_cols)
                new_data = pd.DataFrame(X_pca, columns=pca_new_cols)
                X = pd.concat([X, new_data], axis=1)
            explained_variance = pca.explained_variance_ratio_
            print("NUMBER OF PCA COMPONENTS: ", pca.n_components_)
            print("EXPLAINED VARIANCE OF PCA COMPONENTS: ", explained_variance)

        # Correlation
        drop_features = None
        flag = True
        if y.loc[:, y_ForLearning].astype(float).values.shape[1] == 1:
            y_for_train = y.loc[:, y_ForLearning].astype(float).values.ravel()
        else:
            y_for_train = y.loc[:, y_ForLearning].astype(float).values
        feature_target_corr = feature_target_correlation(X.loc[:, X_ForLearning].astype(float),
                                                         y_for_train,
                                                         np.array(continous_features), PES['PES_MultiTarget'])



        while PES['PES_feat_corr_threshold'] < 1:
            drop_features = correlation(X.loc[:, X_ForLearning], PES['PES_ID'], PES['PES_feat_corr_threshold'], flag)
            flag = False
            if drop_features is None or len(drop_features) == 0:
                break
            worst_feature_to_drop = find_worst_feature(feature_target_corr, drop_features, PES['PES_p_threshold'])
            if worst_feature_to_drop is not None:
                X_ForLearning[X.columns.get_indexer([worst_feature_to_drop])] = False
                print('DROPED FEATURES BECAUSE OF FEATURE CORRELATIONS: ', worst_feature_to_drop)
            else:
                break

        # Only p <= 0.05
        if PES['PES_exclude_gt_p']:
            hypothesis_fails = throw_features_away(feature_target_corr, PES['PES_p_threshold'])
            X_ForLearning[X.columns.get_indexer(hypothesis_fails)] = False
            print('DROPED FEATURES BECAUSE OF P_VALUE > THRESHOLD: ', hypothesis_fails)

        # Insert features in DB (one-hot-encoded and others) for new predictions
        FEA_insert_data(X.loc[:, X_ForLearning].columns.values, feature_target_corr, PES['PES_ID'])
        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=PES['PES_Test_Portion'], shuffle=True)

        # Weight calculations:
        if PES['PES_add_weights']:
            tuple_id_w, weights, left_border, right_border = define_weights(X_train['RATIO'].values,
                                                                            X_train['ID'].values, left_init_cut,
                                                                            right_init_cut, left_cut, right_cut,
                                                                            factor_1, factor_2)
            # X_test = X_test[np.where(np.logical_and(X_test['RATIO'].values>left_border, X_test['RATIO'].values<right_border)), :]
            # y_test = y_test[np.where(np.logical_and(X_test['RATIO'].values>left_border, X_test['RATIO'].values<right_border))]
            TDS_update_weights(tuple_id_w, PES['PES_Table'])
        else:
            weights = []

        # Standardize data and store standardization scaler to drive
        if y_train.shape[0] > 0:
            X_train_st = standardization_train(X_train.loc[:, X_ForLearning].values, PES['PES_ID'], True)
        if y_test.shape[0] > 0:
            X_test_st = standardization_other(X_test.loc[:, X_ForLearning].values, PES['PES_ID'])

        print('NUMBER OF FEATURES TO TRAIN: ', str(len([1 for i in X_ForLearning if i])))

        # Apply PCA
        """if PES['PES_PCA_variance_kept'] < 1.0:
            pca = PCA(n_components=PES['PES_PCA_variance_kept'])
            if y_train.shape[0] > 0:
                X_train_st = pca.fit_transform(X_train_st)
            if y_test.shape[0] > 0:
                X_test_st = pca.transform(X_test_st)
            explained_variance = pca.explained_variance_ratio_
            print("NUMBER OF PCA COMPONENTS: ", pca.n_components_)"""





        """
            BUILD, EVALUATE MODELS
        """
        n_validations = 30
        for i in range(n_validations):
            # Get active models for PES_ID
            TRMS_models_settings = TRMS_select_active_by_pesid(PES['PES_ID'])
            for _, TRMS in TRMS_models_settings.iterrows():
                """if TRMS['TRMS_ID'] <= 1088:
                    continue"""
                print('------------------- Execution start: TRMS_ID = %s -------------------' % TRMS['TRMS_ID'])

                # Get current model
                MLA_model = MLA_select_by_id(TRMS['TRMS_MLA_ID'], PES['PES_MultiTarget'])
                # print(MLA_model['MLA_PythonFunName'][0])
                if MLA_model is None:
                    continue

                # if PES['PES_CrossValidation']:
                #    cross_validation_technique()
                y_test_original = y_test.loc[:, y_ForLearning].values

                y_train_original = y_train.loc[:, y_ForLearning].values

                if y_train.shape[0] > 0:
                    # Build model and store builded object to drive
                    model, feature_importance = build_model(X_train_st, y_train.loc[:, y_ForLearning].values, TRMS,
                                                            MLA_model, weights, PES['PES_MultiTarget'], X_train.columns[X_ForLearning], y_train.columns[y_ForLearning])

                    # Store feature importance
                    if feature_importance.shape[0] > 0:
                        FIM_insert_data(X_train.loc[:, X_ForLearning].columns, feature_importance, TRMS['TRMS_ID'])

                    # Predict train set
                    y_train_pred = predict(X_train_st, TRMS, MLA_model, model)
                    y_train_pred = np.clip(y_train_pred, 0, np.inf)
                    # Evaluate model performance
                    r2_scores_train = r2(y_train_original, y_train_pred,
                                         'raw_values')  # [r2(y_train_original[:, i], y_train_pred[:, i]) for i in range(y_train_pred.shape[1])]

                    if PES['PES_MultiTarget']:
                        rmse_scores_train = [rmse(y_train_original[:, i], y_train_pred[:, i]) for i in
                                         range(y_train_pred.shape[1])]
                    else:
                        rmse_scores_train = [rmse(y_train_original, y_train_pred)]
                    r2_scores_all_train = r2(y_train_original, y_train_pred, 'uniform_average')
                    rmse_scores_all_train = rmse(y_train_original, y_train_pred)
                    TRMS_update_eval_scores(TRMS, r2_scores_train, rmse_scores_train, r2_scores_all_train,
                                            rmse_scores_all_train, PES_TargetVariables, "Train")

                    # Store real and predicted values
                    if PES['PES_MultiTarget']:
                        PRE_insert_multitarget_data(X_train["ID"], y_train.loc[:, y_ForLearning].astype(float).values,
                                                y_train_pred.astype(float), TRMS['TRMS_ID'], 1)
                    else:
                        PRE_insert_single_target_data(X_train["ID"], y_train.loc[:, y_ForLearning].astype(float).values,
                                                y_train_pred.astype(float), TRMS['TRMS_ID'], 1)
                    """except  Exception as e:
                    print(e)
                    continue
                    """

                if y_test.shape[0] > 0:
                    # Predict test set
                    y_test_pred = predict(X_test_st, TRMS, MLA_model)
                    y_test_pred = np.clip(y_test_pred, 0, np.inf)

                    # Evaluate model performance
                    r2_scores_test = r2(y_test_original, y_test_pred,
                                        'raw_values')  # [r2(y_test_original[:,i], y_test_pred[:, i]) for i in range(y_test_pred.shape[1])]
                    if PES['PES_MultiTarget']:
                        rmse_scores_test = [rmse(y_test_original[:, i], y_test_pred[:, i]) for i in range(y_test_pred.shape[1])]
                    else:
                        rmse_scores_test = [rmse(y_test_original, y_test_pred)]

                    r2_scores_all_test = r2(y_test_original, y_test_pred, 'uniform_average')
                    rmse_scores_all_test = rmse(y_test_original, y_test_pred)
                    TRMS_update_eval_scores(TRMS, r2_scores_test, rmse_scores_test, r2_scores_all_test,
                                            rmse_scores_all_test, PES_TargetVariables, "Test")

                    print('R2: ', r2_scores_test)
                    print('RMSE: ', rmse_scores_test)
                    # Store real and predicted values
                    if PES['PES_MultiTarget']:
                        PRE_insert_multitarget_data(X_test["ID"], y_test_original.astype(float), y_test_pred.astype(float),
                                                TRMS['TRMS_ID'], 2)
                    else:
                        PRE_insert_single_target_data(X_test["ID"], y_test_original.astype(float), y_test_pred.astype(float),
                                                TRMS['TRMS_ID'], 2)
                print('------------------- Execution end: TRMS_ID = %s -------------------' % TRMS['TRMS_ID'])
        print(
            '----------------------------------- Execution end: PES_ID = %s -----------------------------------' % PES[
                'PES_ID'])

    """
        2.) PREDICT NEW (UNKNOWN) SAMPLES
    """
    # TODO: - Trenutno se iz PRE izbrišejo zapisi za kombinacijo UNDS_ID, TRMS_ID. V drugem delu se potem napove nove sample in stare še enkrat!!! Bi bilo bolje, da pogledamo, katera kombinacija UNDS_ID, TRMS_ID je že napovedana, in teh ne prediktamo več? Če ja, potem je treba paziti, da ne izbrišem te kombinacije preden insertam
    UNDS_Active_ids = UNDS_select_active_ids()
    for _, UNDS in UNDS_Active_ids.iterrows():
        print('------------------- Execution start: UNDS_ID = %s -------------------' % UNDS['UNDS_ID'])

        TRMS = TRMS_select_by_id(UNDS['UNDS_TRMS_ID'])
        PES = PES_select_by_id(TRMS['TRMS_PES_ID'][0])
        # X = select_data(PES['PES_Table_Unknown'][0])

        PES_OneHotEncodeFeatures = convert_string(PES['PES_OneHotEncodeFeatures'][0])

        if PES_OneHotEncodeFeatures is not None:
            df = one_hot_encoding(X, PES_OneHotEncodeFeatures)

        # Create DataFrame
        X_new = recreate_input_dataset(X, PES['PES_ID'][0])

        if X is None or X.shape[0] == 0:
            continue

        X_unknown_st = standardization_other(X_new, PES['PES_ID'][0])

        MLA_model = MLA_select_by_id(TRMS['TRMS_MLA_ID'], PES['PES_MultiTarget'][0])

        # Predict unknown set
        y_unknown_pred = predict(X_unknown_st, TRMS, MLA_model, None, 3)
        # Store real and predicted values
        PRE_insert_multitarget_data(X["ID"], None, y_unknown_pred, TRMS['TRMS_ID'][0], 3)
        print('----- Execution end: UNDS_ID = %s -----' % UNDS['UNDS_ID'])

if __name__ == '__main__':
    original_stdout = sys.stdout
    #with open(output_file + '\log_output.txt', 'w') as f:
        #sys.stdout = f
    print('-------------------------------------------------- BEGIN AT: %s --------------------------------------------------' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    main()
    print('-------------------------------------------------- END AT: %s --------------------------------------------------' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    #sys.stdout = original_stdout


