from scipy.stats import uniform, randint

LIGHTGBM_PARAMS = {
    'num_leaves': randint(20, 150),
    'max_depth': randint(5, 50),
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(100, 1000),
    'min_child_samples': randint(1, 100),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5)
}

XGBoost_PARAMS = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(5, 50),
    'learning_rate': uniform(0.01, 0.2),
    'min_child_weight': randint(1, 100),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5)
}

CATBOOST_PARAMS = {
    'iterations': randint(100, 1000),
    'depth': randint(5, 50),
    'learning_rate': uniform(0.01, 0.2),
    'l2_leaf_reg': randint(1, 10),
    'border_count': randint(32, 255)
}

RF_PARAMS = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10)
}

RANDOM_SEARCH_PARAMS = {
    'n_iter': 2,
    'scoring': 'f1',
    'cv': 4,
    'verbose': 1,
    'random_state': 42,
    'n_jobs': 4,
    'scoring': 'f1'  # Ensure scoring is set to 'f1'
}