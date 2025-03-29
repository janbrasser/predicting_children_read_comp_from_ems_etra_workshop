import pandas as pd
import numpy as np
import joblib
import datetime


def create_score_results_data_from_file(file_path):
    results = joblib.load(file_path)
    outer_scores_dict = {}
    outer_score_names = [
        'outer_scores_same_year_median_random',
        'outer_scores_same_year_median_grouped',
        'outer_scores_same_year_mean_minus_sd_random',
        'outer_scores_same_year_mean_minus_sd_grouped',
        'outer_scores_next_year_median_random',
        'outer_scores_next_year_median_grouped',
        'outer_scores_next_year_mean_minus_sd_random',
        'outer_scores_next_year_mean_minus_sd_grouped'
    ]

    for score_name in outer_score_names:
        result = np.array(results[score_name])
        mean_score = result.mean()
        sd_score = result.std()
        outer_scores_dict[score_name] = \
            f'Results for {score_name}: {round(mean_score, 3)} +/- {round(sd_score, 3)}'
    return outer_scores_dict


def write_score_results_to_file(results_dict, file_path):
    with open(file_path, 'w') as f:
        for key, value in results_dict.items():
            f.write(f'{value}\n')


def create_feature_importance_results_data_from_file(file_path):
    results = joblib.load(file_path)
    feature_importance_dfs = {}
    best_models_names = [
        'best_models_same_year_median_random',
        'best_models_same_year_median_grouped',
        'best_models_same_year_mean_minus_sd_random',
        'best_models_same_year_mean_minus_sd_grouped',
        'best_models_next_year_median_random',
        'best_models_next_year_median_grouped',
        'best_models_next_year_mean_minus_sd_random',
        'best_models_next_year_mean_minus_sd_grouped'
    ]

    for mode in best_models_names:
        models = results[mode]

        # Extract feature importances
        feature_importances = np.array([model.feature_importances_ for model in models])

        # Compute mean and standard deviation
        mean_importance = np.round(np.mean(feature_importances, axis=0), 3)
        std_importance = np.round(np.std(feature_importances, axis=0),3)

        # Create a dataframe
        feature_names = models[0].feature_names_in_
        df = pd.DataFrame({
            'Feature': feature_names,
            'Mean Importance': mean_importance,
            'Std Importance': std_importance
        })
        # Sort by mean importance
        df_sorted = df.sort_values(by='Mean Importance', ascending=False)
        feature_importance_dfs[mode] = df_sorted

    return feature_importance_dfs


def write_feature_importance_results_to_file(results_dict, filepath):

    with open(filepath, 'w') as f:
        for key, df in results_dict.items():
            f.write(f'{key}\n')
            f.write(f'{df}\n\n')

def main():
    file_path = 'results/best_rf_models.joblib'
    results_dict = create_score_results_data_from_file(file_path)
    write_score_results_to_file(
        results_dict,
        f'results/results_rf_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'
    )
    feature_importance_dfs = create_feature_importance_results_data_from_file(file_path)
    write_feature_importance_results_to_file(
        feature_importance_dfs,
        f'results/feature_importance_rf_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'
    )


if __name__ == '__main__':
    main()
