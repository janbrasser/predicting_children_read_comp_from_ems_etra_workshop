import nam
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

output_dir = 'results/feature_plots'

# load the data
same_year_above_sd = pd.read_csv('training_data/same_year_above_mean-1SD.csv')

next_year_above_sd = pd.read_csv('training_data/next_year_above_mean-1SD.csv')

# define the features to be used
features = ['trial_length', 'total_number_of_fixations', 'total_array_fixation_count',
            'target_fixation_count',
            'target_fixation_rate', 'stimulus_fixation_count', 'average_fixation_duration',
            'average_target_fixation_duration', 'average_non_target_fixation_duration',
            'fixation_duration_variance',
            'refixations', 'refixation_ratio', 'saccades_left', 'saccades_right', 'saccades_on_target_right',
            'saccades_on_target_left', 'saccades_on_target', 'saccades_on_target_ratio',
            'saccade_ratio_right',
            'average_saccade_length', 'saccade_length_variance', 'start_left']

target_column_same_year = 'elfe_bin'
target_column_next_year = 'future_elfe_bin'

hidden_layers = 5
hidden_size = 64
fnet_activation_function = torch.nn.LeakyReLU()
output_activation_function = torch.nn.Sigmoid()
dropout = 0.1
lr = 0.0005
epochs = 50

X_same_year_above_sd = same_year_above_sd[features]
y_same_year_above_sd = same_year_above_sd[target_column_same_year]

X_next_year_above_sd = next_year_above_sd[features]
y_next_year_above_sd = next_year_above_sd[target_column_next_year]

#scale the data
scaler = StandardScaler()
X_same_year_above_sd = scaler.fit_transform(X_same_year_above_sd)
X_next_year_above_sd = scaler.fit_transform(X_next_year_above_sd)

# get the range of the data for plotting later
X_same_year_above_sd_min = X_same_year_above_sd.min(axis=0)[2]
X_same_year_above_sd_max = X_same_year_above_sd.max(axis=0)[2]

X_next_year_above_sd_min = X_next_year_above_sd.min(axis=0)[2]
X_next_year_above_sd_max = X_next_year_above_sd.max(axis=0)[2]

min_value = min(X_same_year_above_sd_min, X_next_year_above_sd_min)
max_value = max(X_same_year_above_sd_max, X_next_year_above_sd_max)

# convert the data to tensors
X_same_year_above_sd_tensor = torch.tensor(X_same_year_above_sd, dtype=torch.float32)
y_same_year_above_sd_tensor = torch.tensor(y_same_year_above_sd, dtype=torch.float32).unsqueeze(1)

X_next_year_above_sd_tensor = torch.tensor(X_next_year_above_sd, dtype=torch.float32)
y_next_year_above_sd_tensor = torch.tensor(y_next_year_above_sd, dtype=torch.float32).unsqueeze(1)

# create the dataloaders
train_data_same_year = torch.utils.data.TensorDataset(X_same_year_above_sd_tensor, y_same_year_above_sd_tensor)
train_loader_same_year = torch.utils.data.DataLoader(train_data_same_year, batch_size=32, shuffle=True)

train_data_next_year = torch.utils.data.TensorDataset(X_next_year_above_sd_tensor, y_next_year_above_sd_tensor)
train_loader_next_year = torch.utils.data.DataLoader(train_data_next_year, batch_size=32, shuffle=True)

# create the model for same year only
model_same_year = nam.Nam(len(features), hidden_layers, hidden_size, fnet_activation_function,
                          output_activation_function, dropout)

if torch.cuda.is_available():
    (print("GPU available, moving model to GPU"))
    model_same_year = model_same_year.cuda()
else:
    print("GPU not available, running on CPU")

# calculate the weights for the loss function
weights = torch.tensor([y_same_year_above_sd.value_counts()[0] / y_same_year_above_sd.value_counts()[1]],
                       dtype=torch.float32)
# put the weights on the GPU if available
if torch.cuda.is_available():
    weights = weights.cuda()

# define the loss function
loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
# define the optimizer
optimizer = torch.optim.Adam(model_same_year.parameters(), lr=lr)

# train the model

# Training loop
for epoch in range(epochs):
    model_same_year.train()
    total_loss = 0

    for batch_X, batch_y in train_loader_same_year:
        if torch.cuda.is_available():
            batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

        optimizer.zero_grad()
        predictions = model_same_year(batch_X)
        loss = loss_function(predictions, batch_y)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader_same_year):.4f}")

examined_features_same_year = ['trial_length', 'total_array_fixation_count', 'total_number_of_fixations',
                               'saccades_right', 'refixations']
examined_features_same_year_rename_dict = {'trial_length': 'Trial Length',
                                           'total_array_fixation_count': 'Total Array Fixation Count',
                                           'total_number_of_fixations': 'Total Number of Fixations',
                                           'saccades_right': 'Saccades Right',
                                           'refixations': 'Refixations'}
# plot the feature distributions
model_same_year.set_feature_names(features)

for feature in examined_features_same_year:
    model_same_year.calculate_feature_net_output_distribution(min_value, max_value,
                                                              0.01, feature_name=feature, plot=True,
                                                              plot_title=
                                                              f'{examined_features_same_year_rename_dict[feature]}',
                                                              plot_title_font_size=20,
                                                              plot_xlabel='Feature Value (normalized)',
                                                              output_file=
                                                              f'{output_dir}/feature_distributions_same_year_'
                                                              f'{feature}.png')

# repeat the process for the next year

# create the model for next year only
model_next_year = nam.Nam(len(features), hidden_layers, hidden_size, fnet_activation_function,
                          output_activation_function, dropout)

if torch.cuda.is_available():
    (print("GPU available, moving model to GPU"))
    model_next_year = model_next_year.cuda()
else:
    print("GPU not available, running on CPU")

# calculate the weights for the loss function
weights = torch.tensor([y_next_year_above_sd.value_counts()[0] / y_next_year_above_sd.value_counts()[1]],
                       dtype=torch.float32)

# put the weights on the GPU if available
if torch.cuda.is_available():
    weights = weights.cuda()

# define the loss function
loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
# define the optimizer
optimizer = torch.optim.Adam(model_next_year.parameters(), lr=lr)

# train the model

# Training loop
for epoch in range(epochs):
    model_next_year.train()
    total_loss = 0

    for batch_X, batch_y in train_loader_next_year:
        if torch.cuda.is_available():
            batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

        optimizer.zero_grad()
        predictions = model_next_year(batch_X)
        loss = loss_function(predictions, batch_y)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader_next_year):.4f}")

examined_features_next_year = ['trial_length', 'average_fixation_duration', 'average_non_target_fixation_duration',
                               'saccades_right', 'saccade_length_variance']

examined_features_next_year_rename_dict = {'trial_length': 'Trial Length',
                                           'average_fixation_duration': 'Average Fixation Duration',
                                           'average_non_target_fixation_duration': 'Average Non-Target Fixation '
                                                                                   'Duration',
                                           'saccades_right': 'Saccades Right',
                                           'saccade_length_variance': 'Saccade Length Variance'}

# plot the feature distributions
model_next_year.set_feature_names(features)

for feature in examined_features_next_year:
    model_next_year.calculate_feature_net_output_distribution(min_value, max_value,
                                                              0.01, feature_name=feature, plot=True,
                                                              plot_title=
                                                              f'{examined_features_next_year_rename_dict[feature]}',
                                                              plot_title_font_size=20,
                                                              plot_xlabel='Feature Value (normalized)',
                                                              output_file=
                                                              f'{output_dir}/feature_distributions_next_year_'
                                                              f'{feature}.png')
