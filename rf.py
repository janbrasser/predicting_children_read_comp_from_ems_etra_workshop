import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (cross_val_score, RandomizedSearchCV, GridSearchCV, train_test_split, KFold,
                                     StratifiedGroupKFold)
import os
import joblib


def cross_validate_rf(model, X, y, param_grid, inner_cv, outer_cv, group=None):
    best_models = []
    outer_scores = []

    if group:
        outer_split = outer_cv.split(X, y, groups=X[group])
    else:
        outer_split = outer_cv.split(X, y)

    for i, (train_idx, test_idx) in enumerate(outer_split):
        print(f"Fold {i + 1} of {outer_cv.get_n_splits()}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        X_train_numeric = X_train.drop(columns=['participant_id'])
        X_test_numeric = X_test.drop(columns=['participant_id'])

        grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1, verbose=1)

        if group:
            grid_search.fit(X_train_numeric, y_train, groups=X_train[group])
        else:
            grid_search.fit(X_train_numeric, y_train)

        best_models.append(grid_search.best_estimator_)
        best_score = grid_search.best_estimator_.score(X_test_numeric, y_test)
        outer_scores.append(best_score)

        print(f"Fold completed. Best params: {grid_search.best_params_}, Test Accuracy: {best_score:.4f}")

    return best_models, outer_scores


def main():
    data_directory = 'data/training_data'

    # Load the data

    df_same_year_median = pd.read_csv(os.path.join(data_directory, 'same_year_above_median.csv'))
    df_same_year_mean_minus_SD = pd.read_csv(os.path.join(data_directory, 'same_year_above_mean-1SD.csv'))
    df_next_year_median = pd.read_csv(os.path.join(data_directory, 'next_year_above_median.csv'))
    df_next_year_mean_minus_SD = pd.read_csv(os.path.join(data_directory, 'next_year_above_mean-1SD.csv'))

    # define the features and the target
    features = ['trial_length', 'total_number_of_fixations', 'total_array_fixation_count', 'target_fixation_count',
                'target_fixation_rate', 'stimulus_fixation_count', 'average_fixation_duration',
                'average_target_fixation_duration', 'average_non_target_fixation_duration',
                'fixation_duration_variance',
                'refixations', 'refixation_ratio', 'saccades_left', 'saccades_right', 'saccades_on_target_right',
                'saccades_on_target_left', 'saccades_on_target', 'saccades_on_target_ratio', 'saccade_ratio_right',
                'average_saccade_length', 'saccade_length_variance', 'start_left', 'participant_id']
    target_same_year = 'elfe_bin'
    target_next_year = 'future_elfe_bin'

    X_same_year_median = df_same_year_median[features]
    y_same_year_median = df_same_year_median[target_same_year]
    X_same_year_mean_minus_SD = df_same_year_mean_minus_SD[features]
    y_same_year_mean_minus_SD = df_same_year_mean_minus_SD[target_same_year]
    X_next_year_median = df_next_year_median[features]
    y_next_year_median = df_next_year_median[target_next_year]
    X_next_year_mean_minus_SD = df_next_year_mean_minus_SD[features]
    y_next_year_mean_minus_SD = df_next_year_mean_minus_SD[target_next_year]

    # define the hyperparameter search space
    param_grid = {'n_estimators': [50, 100, 150, 200],
                  'max_features': ['log2', 'sqrt'],
                  'max_depth': [3, 5, 10, 20, None],
                  'min_samples_split': [2, 5, 10],
                  'min_samples_leaf': [1, 2, 4, 10, 50]}

    # Outer cross-validation (for performance evaluation)
    outer_cv_grouped = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    outer_cv_random = KFold(n_splits=5, shuffle=True, random_state=42)

    # Inner cross-validation (for hyperparameter tuning)
    inner_cv_grouped = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv_random = KFold(n_splits=5, shuffle=True, random_state=42)

    # define the model
    rf = RandomForestClassifier()

    # store results
    results = {}

    # perform nested cross validation for each dataset
    results['best_models_same_year_median_random'], results['outer_scores_same_year_median_random'] = (
        cross_validate_rf(rf, X_same_year_median, y_same_year_median, param_grid, inner_cv_random, outer_cv_random))
    results['best_models_same_year_median_grouped'], results['outer_scores_same_year_median_grouped'] = (
        cross_validate_rf(rf, X_same_year_median, y_same_year_median, param_grid, inner_cv_grouped, outer_cv_grouped,
                          group='participant_id'))
    results['best_models_same_year_mean_minus_SD_random'], results['outer_scores_same_year_mean_minus_SD_random'] = (
        cross_validate_rf(rf, X_same_year_mean_minus_SD, y_same_year_mean_minus_SD, param_grid, inner_cv_random,
                          outer_cv_random))
    results['best_models_same_year_mean_minus_SD_grouped'], results['outer_scores_same_year_mean_minus_SD_grouped'] = (
        cross_validate_rf(rf, X_same_year_mean_minus_SD, y_same_year_mean_minus_SD, param_grid, inner_cv_grouped,
                          outer_cv_grouped,
                          group='participant_id'))
    results['best_models_next_year_median_random'], results['outer_scores_next_year_median_random'] = (
        cross_validate_rf(rf, X_next_year_median, y_next_year_median, param_grid, inner_cv_random, outer_cv_random))
    results['best_models_next_year_median_grouped'], results['outer_scores_next_year_median_grouped'] = (
        cross_validate_rf(rf, X_next_year_median, y_next_year_median, param_grid, inner_cv_grouped, outer_cv_grouped,
                          group='participant_id'))
    results['best_models_next_year_mean_minus_SD_random'], results['outer_scores_next_year_mean_minus_SD_random'] = (
        cross_validate_rf(rf, X_next_year_mean_minus_SD, y_next_year_mean_minus_SD, param_grid, inner_cv_random,
                          outer_cv_random))
    results['best_models_next_year_mean_minus_SD_grouped'], results['outer_scores_next_year_mean_minus_SD_grouped'] = (
        cross_validate_rf(rf, X_next_year_mean_minus_SD, y_next_year_mean_minus_SD, param_grid, inner_cv_grouped,
                          outer_cv_grouped,
                          group='participant_id'))

    # save the best models to a file
    joblib.dump(results, 'results/best_rf_models.joblib')


if __name__ == '__main__':
    main()
