from sklearn.impute import SimpleImputer
from db_read_write import COR_insert_correlations, FEA_select_by_pes_id
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from db_connection import model_storage
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.linear_model import Ridge, MultiTaskElasticNet, MultiTaskLasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import f_regression, f_classif, VarianceThreshold
from scipy.stats import norm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import  decimal

"""
---------------------------  GROUP: DATA TRANSFORMATIONS  ---------------------------
"""

def recreate_input_dataset(X, PES_ID):
    FEA_learned_features = FEA_select_by_pes_id(PES_ID)['FEA_FeatureName'].values
    output = np.zeros(shape=(X.shape[0], FEA_learned_features.shape[0]))
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if FEA_learned_features[j] in X.columns:
                output[i, j] = X[FEA_learned_features[j]][i]
    return output

def one_hot_encoding(df, categorical_features):
    return pd.get_dummies(df, columns=categorical_features, drop_first=False)

def impute_missing_values(df, features_to_impute, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = df.copy()
    df_imputed[features_to_impute] = imputer.fit_transform(df[features_to_impute])
    return df_imputed

def standardization_train(data, PES_ID, store):
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    result = scaler.fit_transform(data)
    if store:
        joblib.dump(scaler, '%s/PES_ID_%s_scaler.pkl' % (model_storage, PES_ID), compress=9)
    return result

def standardization_other(data, PES_ID):
    scaler = joblib.load('%s/PES_ID_%s_scaler.pkl' % (model_storage, PES_ID))
    result = scaler.transform(data)
    return result

def correlation_prepare_for_insert(matrix, PES_ID):
    result = []
    for (FeatureName1, data) in matrix.items():
        for (FeatureName2, value) in data.dropna().items():
            result.append((PES_ID, FeatureName1, FeatureName2, value))
    return result

def identify_high_corr_features(corr_matrix, threshold):
    to_drop = np.full((corr_matrix.shape[0],), False, dtype=bool)
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[0]):
            if corr_matrix.iloc[i, j] >= threshold:
                # check if i or j is already in to_drop
                if not to_drop[j]:
                    to_drop[j] = True
                if not to_drop[i]:
                    to_drop[i] = True
    return to_drop

def correlation(data, PES_ID, PES_threshold, insert = False):
    cor_matrix = data.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    to_drop = identify_high_corr_features(cor_matrix, PES_threshold)#[column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
    if insert:
        COR_insert_correlations(correlation_prepare_for_insert(upper_tri, PES_ID), PES_ID)
    return data.columns[to_drop].values


def eliminate_constants(df, threshold = 0):
    constant_filter = VarianceThreshold(threshold=threshold)
    constant_filter.fit(df)
    constant_columns = [column for column in df.columns
                        if column not in df.columns[constant_filter.get_support()]]
    return constant_columns

def feature_target_correlation(X, y, continous_features, multitarget):
    X_continous = X.loc[:, X.columns.isin(continous_features)]
    X_categorical = X.loc[:, ~X.columns.isin(continous_features)]
    if multitarget == 0:
        if X_continous.shape[1] > 0:
            F_values_con, p_values_con = f_regression(X_continous, y)
        if X_categorical.shape[1] > 0:
            F_values_cat, p_values_cat = f_classif(X_categorical, y)
    else:
        if X_continous.shape[1] > 0:
            F_values1, p_values1 = f_regression(X_continous, y[:, 0])
            F_values2, p_values2 = f_regression(X_continous, y[:, 1])
            F_values_con = (F_values1 + F_values2) / 2.0
            p_values_con = (p_values1 + p_values2) / 2.0
        if X_categorical.shape[1] > 0:
            F_values1, p_values1 = f_classif(X_categorical, y[:, 0])
            F_values2, p_values2 = f_classif(X_categorical, y[:, 1])
            F_values_cat = (F_values1 + F_values2) / 2.0
            p_values_cat = (p_values1 + p_values2) / 2.0
    selected_features = []
    for i in range(X_continous.shape[1]):
        F_values_con[np.where(F_values_con > 1000000)] = 1000000
        selected_features.append({'feature': X_continous.columns[i], 'F-value': F_values_con[i], 'p-value': p_values_con[i]})
    for i in range(X_categorical.shape[1]):
        F_values_cat[np.where(F_values_cat > 1000000)] = 1000000
        selected_features.append({'feature': X_categorical.columns[i], 'F-value': F_values_cat[i], 'p-value': p_values_cat[i]})
    # sort the selected features based on their F-test score in descending order
    selected_features.sort(key=lambda x: x['p-value'], reverse=True)
    return selected_features


def find_worst_feature(selected_features, drop_features, p_threshold):
    worst_score = 0
    worst_feature = None
    for i, feature in enumerate(selected_features):
        if feature['feature'] in drop_features:
            if feature['p-value'] > worst_score:
                worst_score = feature['p-value']
                worst_feature = feature['feature']
    #if worst_feature is None:
        #worst_feature = drop_features[0]
    return worst_feature


def throw_features_away(selected_features, threshold):
    features = []
    for i, feature in enumerate(selected_features):
        if feature['p-value'] > threshold:
            features.append(feature['feature'])
    return features
"""
---------------------------  GROUP: BUILD MODEL  ---------------------------
"""


def build_model(X, y, TRMS, MLA_model, weights, multitarget, X_cols = None, y_cols = None):
    if not multitarget:
        y = y.ravel()
    model = None
    if TRMS['TRMS_JSONParams'] is not None:
        model_parameters = json.loads(TRMS['TRMS_JSONParams'])
    else:
        model_parameters = {}
    #print(TRMS)
    if MLA_model['MLA_PythonFunName'][0] == 'RandomForestRegressor':
        # Build model

        if MLA_model['MLA_MultiTarget'][0]:
            max_features = 1
            max_depth = None
            n_est = None
            if 'n_estimators' in model_parameters:
                n_est = model_parameters['n_estimators']
            if 'max_features' in model_parameters:
                max_features = model_parameters['max_features']
            if 'max_depth' in model_parameters:
                max_depth = model_parameters['max_depth']
            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_est,  max_features = max_features, max_depth=max_depth))
            if len(weights) > 0:
                model.fit(X, y, sample_weight = weights,)
            else:
                model.fit(X, y)
            # Store model
            store_model(model, TRMS['TRMS_ID'])
            importance = transform_importance(model.estimators_, 'feature_importances_')
        else:
            max_features = 1
            max_depth = None
            n_est = None
            if 'n_estimators' in model_parameters:
                n_est = model_parameters['n_estimators']
            if 'max_features' in model_parameters:
                max_features = model_parameters['max_features']
            if 'max_depth' in model_parameters:
                max_depth = model_parameters['max_depth']
            model = RandomForestRegressor(n_estimators=n_est, max_features=max_features, max_depth=max_depth)
            if len(weights) > 0:
                model.fit(X, y, sample_weight=weights, )
            else:
                model.fit(X, y)
            # Store model
            store_model(model, TRMS['TRMS_ID'])
            importance = transform_importance(model.estimators_, 'feature_importances_')

    if MLA_model['MLA_PythonFunName'][0] == 'LinearRegressor':
        # MultiTarget:
        if MLA_model['MLA_MultiTarget'][0]:
            if 'degree' in model_parameters and model_parameters['degree'] > 1:
                degree = model_parameters['degree']
                poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
                X = poly.fit_transform(X)
                joblib.dump(poly, '%s/TRMS_ID_%s_poly.pkl' % (model_storage, TRMS['TRMS_ID']), compress=9)
            model = LinearRegression()
            model.fit(X, y)
            store_model(model, TRMS['TRMS_ID'])
            importance = np.array([])#transform_importance(model.estimators_, "coef_")
        else:
            return None, None
    if MLA_model['MLA_PythonFunName'][0] == 'SVR':
        # MultiTarget:
        if MLA_model['MLA_MultiTarget'][0]:
            epsilon = 0.1
            C = 1
            degree = 3
            kernel = 'rbf'
            if 'kernel' in model_parameters:
                kernel = model_parameters['kernel']
            if 'epsilon' in model_parameters:
                epsilon = model_parameters['epsilon']
            if 'C' in model_parameters:
                C = model_parameters['C']
            if 'degree' in model_parameters:
                degree = model_parameters['degree']
            model = MultiOutputRegressor(SVR(C=C, epsilon=epsilon, degree=degree, kernel=kernel)) # , max_iter=1000
            model.fit(X, y)
            store_model(model, TRMS['TRMS_ID'])
            results = permutation_importance(model, X, y, scoring='neg_mean_squared_error')
            # get importance
            importance = transform_importance(results, "importances_mean")
        else:
            return None, None

    if MLA_model['MLA_PythonFunName'][0] == 'Ridge':
        # MultiTarget:
        if MLA_model['MLA_MultiTarget'][0]:
            model = MultiOutputRegressor(Ridge(alpha=model_parameters['alpha']))
            model.fit(X, y)
            store_model(model, TRMS['TRMS_ID'])
            importance = transform_importance(model.estimators_, "coef_")
        else:
            return None, None

    if MLA_model['MLA_PythonFunName'][0] == 'GradientBoostingRegressor':
        # MultiTarget:
        if MLA_model['MLA_MultiTarget'][0]:
            max_features = 1
            max_depth = 5
            n_est = None
            if 'n_estimators' in model_parameters:
                n_est = model_parameters['n_estimators']
            if 'max_features' in model_parameters:
                max_features = model_parameters['max_features']
            if 'max_depth' in model_parameters:
                max_depth = model_parameters['max_depth']
            model = MultiOutputRegressor(estimator=GradientBoostingRegressor(n_estimators=n_est,  max_features = max_features, max_depth=max_depth))
            if len(weights) > 0:
                model.fit(X, y, sample_weight = weights)
            else:
                model.fit(X, y)
            store_model(model, TRMS['TRMS_ID'])
            importance = transform_importance(model.estimators_, "feature_importances_")
        else:
            max_features = 1
            max_depth = 5
            n_est = None
            if 'n_estimators' in model_parameters:
                n_est = model_parameters['n_estimators']
            if 'max_features' in model_parameters:
                max_features = model_parameters['max_features']
            if 'max_depth' in model_parameters:
                max_depth = model_parameters['max_depth']
            model = GradientBoostingRegressor(n_estimators=n_est, max_features=max_features, max_depth=max_depth)
            if len(weights) > 0:
                model.fit(X, y, sample_weight=weights)
            else:
                model.fit(X, y)
            store_model(model, TRMS['TRMS_ID'])
            #print(model.estimators_)
            #importance = transform_importance(model.estimators_, "feature_importances_")
            importance = np.array([])




    if MLA_model['MLA_PythonFunName'][0] == 'MultiTaskLasso':
        # Build model
        if MLA_model['MLA_MultiTarget'][0]:
            alpha = 1.0
            if 'alpha' in model_parameters:
                alpha = model_parameters['alpha']
            max_iter = 1000
            if 'max_iter' in model_parameters:
                max_iter = model_parameters['max_iter']
            model = MultiTaskLasso(alpha=alpha, max_iter=max_iter)# , max_features=max_features max_features = 0.3
            model.fit(X, y)

            # Store model
            store_model(model, TRMS['TRMS_ID'])
            #importance = transform_importance(model.estimators_, 'coef_')
            importance = np.array([])
        else:
            return None, None

    if MLA_model['MLA_PythonFunName'][0] == 'MultiTaskElasticNet':
        # Build model
        if MLA_model['MLA_MultiTarget'][0]:
            alpha = 1.0
            if 'alpha' in model_parameters:
                alpha = model_parameters['alpha']
            max_iter = 1000
            if 'max_iter' in model_parameters:
                max_iter = model_parameters['max_iter']
            model = MultiTaskElasticNet(alpha=alpha, max_iter=max_iter)# , max_features=max_features max_features = 0.3
            model.fit(X, y)

            # Store model
            store_model(model, TRMS['TRMS_ID'])
            #importance = transform_importance(model.estimators_, 'coef_')
            importance = np.array([])
        else:
            return None, None

    if MLA_model['MLA_PythonFunName'][0] == 'MLPRegressor':
        # Build model
        if MLA_model['MLA_MultiTarget'][0]:
            solver = 'adam'
            if 'solver' in model_parameters:
                solver = model_parameters['solver']
            max_iter = 200
            if 'max_iter' in model_parameters:
                max_iter = model_parameters['max_iter']
            if 'hidden_layer_sizes' in model_parameters:
                hidden_layer_sizes = model_parameters['max_iter']
            else:
                hidden_layer_sizes = (100,)
            model = MLPRegressor(hidden_layer_sizes= hidden_layer_sizes, max_iter=max_iter, solver=solver)  # , max_features=max_features max_features = 0.3

            #model = MultiOutputRegressor(MLPRegressor(max_iter = max_iter, solver=solver)) # , max_features=max_features max_features = 0.3
            model.fit(X, y)

            # Store model
            store_model(model, TRMS['TRMS_ID'])
            #importance = transform_importance(model.estimators_, 'coef_')
            importance = np.array([])
        else:
            solver = 'adam'
            if 'solver' in model_parameters:
                solver = model_parameters['solver']
            max_iter = 200
            if 'max_iter' in model_parameters:
                max_iter = model_parameters['max_iter']
            if 'hidden_layer_sizes' in model_parameters:
                hidden_layer_sizes = model_parameters['max_iter']
            else:
                hidden_layer_sizes = (100,)
            model = MLPRegressor(hidden_layer_sizes= hidden_layer_sizes, max_iter=max_iter, solver=solver)  # , max_features=max_features max_features = 0.3
            model.fit(X, y)

            # Store model
            store_model(model, TRMS['TRMS_ID'])
            # importance = transform_importance(model.estimators_, 'coef_')
            importance = np.array([])



    if MLA_model['MLA_PythonFunName'][0] == 'KNeighborsRegressor':
        # Build model
        if MLA_model['MLA_MultiTarget'][0]:
            model = MultiOutputRegressor(KNeighborsRegressor()) # , max_features=max_features max_features = 0.3
            model.fit(X, y)

            # Store model
            store_model(model, TRMS['TRMS_ID'])
            #importance = transform_importance(model.estimators_, 'coef_')
            importance = np.array([])
        else:
            return None, None


    if MLA_model['MLA_PythonFunName'][0] == 'GaussianProcessRegressor':
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        # Create a Gaussian Process regressor with a kernel that models the correlation between targets
        if 'length_scale' in model_parameters:
            length_scale = model_parameters['length_scale']
        else:
            length_scale = 1.0
        kernel = RBF(length_scale=length_scale)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(X, y)

        # Predict the target variables on the test set

        store_model(model, TRMS['TRMS_ID'])
        # importance = transform_importance(model.estimators_, 'coef_')
        importance = np.array([])
    if MLA_model['MLA_PythonFunName'][0] == 'BayesianEstimator':
        from pgmpy.models import BayesianModel
        from pgmpy.estimators import BayesianEstimator
        # Create a Gaussian Process regressor with a kernel that models the correlation between targets
        model = BayesianModel()
        model.add_nodes_from(X_cols)
        model.add_nodes_from(y_cols)
        # Predict the target variables on the test set
        #est = BayesianEstimator(model, X, y)
        model.fit(X, y, estimator=BayesianEstimator)
        #model.fit(est)
        store_model(model, TRMS['TRMS_ID'])
        # importance = transform_importance(model.estimators_, 'coef_')
        importance = np.array([])
    if MLA_model['MLA_PythonFunName'][0] == 'VotingRegressor':
        # Build model
        if MLA_model['MLA_MultiTarget'][0]:
            max_features = 1
            max_depth = None
            n_est = None
            solver = 'adam'
            if 'solver' in model_parameters:
                solver = model_parameters['solver']
            if 'n_estimators' in model_parameters:
                n_est = model_parameters['n_estimators']
            if 'max_features' in model_parameters:
                max_features = model_parameters['max_features']
            if 'max_depth' in model_parameters:
                max_depth = model_parameters['max_depth']
            epsilon = 0.1
            C = 1
            degree = 3
            kernel = 'rbf'
            if 'kernel' in model_parameters:
                kernel = model_parameters['kernel']
            if 'epsilon' in model_parameters:
                epsilon = model_parameters['epsilon']
            if 'C' in model_parameters:
                C = model_parameters['C']
            if 'degree' in model_parameters:
                degree = model_parameters['degree']

            max_iter = 1000
            if 'max_iter' in model_parameters:
                max_iter = model_parameters['max_iter']
            #svr = MultiOutputRegressor(SVR(C=C, epsilon=epsilon, degree=degree, kernel=kernel))
            rf = RandomForestRegressor(n_estimators=n_est,  max_features = max_features) # , max_features=max_features max_features = 0.3 max_depth=max_depth,
            lr = LinearRegression()
            nn = MLPRegressor(max_iter = max_iter, solver=solver)
            kNN = KNeighborsRegressor()
            grad = GradientBoostingRegressor(n_estimators=n_est, max_features=max_features, max_depth=max_depth)
            model = MultiOutputRegressor(VotingRegressor([('rf', rf), ('lr', lr), ('nn', nn), ('kNN', kNN), ('grad', grad)])) #('svr', svr)

            model.fit(X, y)
            # Store model
            store_model(model, TRMS['TRMS_ID'])
            importance = np.array([])#transform_importance(model.estimators_, 'feature_importances_')
        else:
            return None, None

    return model, importance









"""
---------------------------  GROUP: PREDICT AND EVALUATE MODEL  ---------------------------  
"""

def predict(X, TRMS, MLA_model, model = None, type=1):

    predictions = None

    # Load model if exists
    if model is None:
        if type == 3:
            model = load_model(TRMS['TRMS_ID'][0])
        else:
            model = load_model(TRMS['TRMS_ID'])
        if model is None:
            return None

    # Predict based on model name
    if MLA_model['MLA_PythonFunName'][0] == 'LinearRegressor':
        model_parameters = {}
        if TRMS['TRMS_JSONParams'] is not None:
            model_parameters = json.loads(TRMS['TRMS_JSONParams'])
        if 'degree' in model_parameters and model_parameters['degree'] > 1:
            poly = joblib.load('%s/TRMS_ID_%s_poly.pkl' % (model_storage, TRMS['TRMS_ID']))
            X = poly.transform(X)
        predictions = model.predict(X)
    else:
        predictions = model.predict(X)
    return predictions









"""
---------------------------  GROUP: SCORING FUNCTIONS  ---------------------------
"""

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2(y_true, y_pred, multi):
    return r2_score(y_true, y_pred, multioutput=multi)

def aic(y, y_pred, k):
    n = len(y)
    residual = np.sum((y - y_pred)**2)
    aic = n * np.log(residual/n) + 2*k
    return aic

def bic(y, y_pred, k):
    n = len(y)
    residual = np.sum((y - y_pred)**2)
    bic = n * np.log(residual/n) + np.log(n)*k
    return bic






def custom_loss(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    neg_penalty = 100 if mse < 100 else mse
    print(mse if (y_pred >= 0) else neg_penalty)
    return mse if (y_pred >= 0) else neg_penalty



"""
---------------------------  GROUP: OTHER  ---------------------------
"""


def chebyshev_inequality(k):
    return 1 / k**2

def transform_importance(arr, fun_name):
    res = ()
    for i in range(len(arr)):
        if fun_name == 'feature_importances_':
            res = (*res, (arr[i].feature_importances_))
        elif fun_name == 'coef_':
            res = (*res, (arr[i].coef_))
        elif fun_name == 'importances_mean':
            res = (*res, (arr[i].importances_mean))

    return np.vstack(res).T


def convert_string(string):
    if string is None:
        return None
    return list(map((lambda x: x), string.replace("[", "")[:-1].split("]")))

def store_model(model, TRMS_ID):
    joblib.dump(model, '%s/TRMS_ID_%s_model.pkl' % (model_storage, TRMS_ID), compress=9)

def load_model(TRMS_ID):
    try:
        model = joblib.load('%s/TRMS_ID_%s_model.pkl' % (model_storage, TRMS_ID))
        return model
    except:
        return None


def define_weights(data, ids, left_init_cut, right_init_cut, left_cut, right_cut, factor, second_factor):
    data = data.astype(float)

    original_data = data.copy()
    data = data[(data > left_init_cut) & (data < right_init_cut)]

    # ORIGINAL
    mean = np.mean(data)
    std_dev = np.std(data)

    modified_data = data
    if left_cut:
        modified_data = modified_data[(modified_data > mean - factor * std_dev)]

    if right_cut:
        modified_data = modified_data[(modified_data < mean + factor * std_dev)]

    mean = np.mean(modified_data)
    std_dev = np.std(modified_data)


    left_border = mean - second_factor * std_dev
    right_border = mean + second_factor * std_dev
    print(left_border, right_border)

    data_weights = norm.pdf(original_data, mean, std_dev)
    #data_weights[np.where(np.logical_and(original_data>left_border, original_data<right_border))] = 1
    return list(zip(ids, data_weights)), np.array(data_weights), left_border, right_border
