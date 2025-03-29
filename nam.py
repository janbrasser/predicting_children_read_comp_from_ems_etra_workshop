"""
Implementation of neural additive models for predicting word level reading comprehension in children from eye-tracking
data on a visual search task.
"""

import torch
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, KFold
from sklearn.preprocessing import StandardScaler


class FeatureNet(torch.nn.Module):
    def __init__(self, hidden_layers, hidden_size, activation_function=torch.nn.ReLU(), dropout=0.2):
        super(FeatureNet, self).__init__()

        self.input_layer = torch.nn.Linear(1, hidden_size)
        self.hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size, bias=True) for _ in range(hidden_layers)])
        self.output_layer = torch.nn.Linear(hidden_size, 1)
        self.activation_function = activation_function
        self.dropout = torch.nn.Dropout(dropout)

        self.feature_name = None

    def forward(self, x):
        x = self.activation_function(self.input_layer(x))
        x = self.dropout(x)
        for hidden_layer in self.hidden_layers:
            x = self.activation_function(hidden_layer(x))
            x = self.dropout(x)
        return self.output_layer(x)

    def set_feature_name(self, feature_name):
        self.feature_name = feature_name


class Nam(torch.nn.Module):
    def __init__(self,
                 num_features,
                 hidden_layers,
                 hidden_size,
                 fnet_activation_function=torch.nn.ReLU(),
                 output_activation_function=torch.nn.Sigmoid(),
                 dropout=0.2):
        super(Nam, self).__init__()

        self.feature_nets = torch.nn.ModuleList(
            [FeatureNet(hidden_layers, hidden_size, fnet_activation_function, dropout) for _ in range(num_features)]
        )
        self.output_layer = torch.nn.Linear(num_features, 1, bias=True)
        self.output_activation_function = output_activation_function
        self.feature_names = None

    def forward(self, x):
        x = torch.cat([feature_net(x[:, i].unsqueeze(1)) for i, feature_net in enumerate(self.feature_nets)], dim=1)
        return self.output_activation_function(self.output_layer(x))

    def train_nam(self, x, y, epochs, loss_function, optimizer=None):
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self(x)
            loss = loss_function(y_pred, y)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def set_feature_names(self, feature_names):
        self.feature_names = feature_names
        for feature_net, feature_name in zip(self.feature_nets, feature_names):
            feature_net.set_feature_name(feature_name)

    def calculate_feature_net_output(self, x, feature_net_index=None, feature_name=None):

        if feature_net_index is not None and feature_name is not None:
            print("Please provide either feature_net_index or feature_name but not both.")
            return None

        if feature_net_index is not None:
            return self.feature_nets[feature_net_index](torch.tensor([x]))
        elif feature_name is not None:
            for net in self.feature_nets:
                if net.feature_name == feature_name:
                    return net(torch.tensor([x]))
            print(f"Feature name {feature_name} not found.")
            return None
        else:
            print("Please provide either feature_net_index or feature_name.")
            return None

    def calculate_feature_net_output_distribution(self, min_val, max_val, step,
                                                  feature_net_index=None, feature_name=None, plot=False,
                                                  y_min=None, y_max=None, plot_title=None, plot_title_font_size=None,
                                                  plot_xlabel=None, plot_ylabel=None, output_file=None):

        if feature_net_index is not None and feature_name is not None:
            print("Please provide either feature_net_index or feature_name but not both.")
            return None

        if feature_net_index is not None:
            idx = feature_net_index
        elif feature_name is not None:
            for i, net in enumerate(self.feature_nets):
                if net.feature_name == feature_name:
                    idx = i
                    break
            else:
                print(f"Feature name {feature_name} not found.")
                return None
        else:
            print("Please provide either feature_net_index or feature_name.")
            return None

        feature_net = self.feature_nets[idx]
        x = torch.arange(min_val, max_val, step).unsqueeze(1)
        if torch.cuda.is_available():
            x = x.cuda()
            feature_net = feature_net.cuda()
        y = feature_net(x)
        y = y.cpu().detach().numpy()
        x = x.cpu().numpy()

        if plot:
            plt.clf()
            if y_min is not None and y_max is not None:
                plt.ylim(y_min, y_max)
            plt.plot(x, y)
            if plot_xlabel is not None:
                plt.xlabel(plot_xlabel)
            else:
                plt.xlabel("Feature value")
            if plot_ylabel is not None:
                plt.ylabel(plot_ylabel)
            else:
                plt.ylabel("Feature net output")
            if plot_title is not None:
                plt.title(plot_title)
            else:
                plt.title(f"Feature net output distribution for {feature_name}")
            if plot_title_font_size is not None:
                plt.title(plot_title, fontsize=plot_title_font_size)

            if output_file is not None:
                plt.savefig(output_file)
            else:
                plt.show()

        return y


def main():
    # create output file
    output_file = open('results/nam_results.txt', 'w')
    # load training data
    same_year_above_median = pd.read_csv('training_data/same_year_above_med.csv')
    same_year_above_sd = pd.read_csv('training_data/same_year_above_mean-1SD.csv')
    next_year_above_median = pd.read_csv('training_data/next_year_above_median.csv')
    next_year_above_sd = pd.read_csv('training_data/next_year_above_mean-1SD.csv')

    # define the features to be used
    features = features = ['trial_length', 'total_number_of_fixations', 'total_array_fixation_count',
                           'target_fixation_count',
                           'target_fixation_rate', 'stimulus_fixation_count', 'average_fixation_duration',
                           'average_target_fixation_duration', 'average_non_target_fixation_duration',
                           'fixation_duration_variance',
                           'refixations', 'refixation_ratio', 'saccades_left', 'saccades_right',
                           'saccades_on_target_right',
                           'saccades_on_target_left', 'saccades_on_target', 'saccades_on_target_ratio',
                           'saccade_ratio_right',
                           'average_saccade_length', 'saccade_length_variance', 'start_left']
    participant_id_column = 'participant_id'
    target_column_same_year = 'elfe_bin'
    target_column_next_year = 'future_elfe_bin'

    # define the hyperparameters
    hidden_layers = 5
    hidden_size = 64
    fnet_activation_function = torch.nn.LeakyReLU()
    output_activation_function = torch.nn.Sigmoid()
    dropout = 0.1
    lr = 0.0005
    epochs = 50

    # train and evaluate the model for same year above median
    print("Training model for same year above median random split")

    # set up the training data
    X_same_year_median = same_year_above_median[features]
    y_same_year_median = same_year_above_median[target_column_same_year]
    groups_same_year_median = same_year_above_sd[participant_id_column]

    X_next_year_median = next_year_above_median[features]
    y_next_year_median = next_year_above_median[target_column_next_year]
    groups_next_year_median = next_year_above_median[participant_id_column]

    X_same_year_above_sd = same_year_above_sd[features]
    y_same_year_above_sd = same_year_above_sd[target_column_same_year]
    groups_same_year_above_sd = same_year_above_sd[participant_id_column]

    X_next_year_above_sd = next_year_above_sd[features]
    y_next_year_above_sd = next_year_above_sd[target_column_next_year]
    groups_next_year_above_sd = next_year_above_sd[participant_id_column]

    # make the splits for the cross validation
    cv_random = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_grouped = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores = []
    # train and evaluate the model
    for fold, (train_index, test_index) in enumerate(cv_random.split(X_same_year_median, y_same_year_median)):
        X_train, X_test = X_same_year_median.iloc[train_index], X_same_year_median.iloc[test_index]
        y_train, y_test = y_same_year_median.iloc[train_index], y_same_year_median.iloc[test_index]

        # standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

        # create dataloaders
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

        test_data = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

        # initialize the model
        model_same_year_median = Nam(len(features), hidden_layers, hidden_size, fnet_activation_function,
                                     output_activation_function, dropout)

        if torch.cuda.is_available():
            (print("GPU available, moving model to GPU"))
            model_same_year_median = model_same_year_median.cuda()
        else:
            print("GPU not available, running on CPU")

        #calculate the weights for the loss function
        weights = torch.tensor([y_same_year_median.value_counts()[0] / y_same_year_median.value_counts()[1]],
                               dtype=torch.float32)
        # put the weights on the GPU if available
        if torch.cuda.is_available():
            weights = weights.cuda()

        # define the loss function
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        # define the optimizer
        optimizer = torch.optim.Adam(model_same_year_median.parameters(), lr=lr)

        print(f'Number of model parameters: {sum(p.numel() for p in model_same_year_median.parameters())}')
        # Training loop
        for epoch in range(epochs):
            model_same_year_median.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                optimizer.zero_grad()
                predictions = model_same_year_median(batch_X)
                loss = loss_function(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

            # Validation phase
            model_same_year_median.eval()
            val_losses = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    if torch.cuda.is_available():
                        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                    predictions = model_same_year_median(batch_X)  # Probabilities from Sigmoid
                    val_losses.append(loss_function(predictions, batch_y).item())

            print(f"Fold {fold + 1} Validation Loss: {sum(val_losses) / len(val_losses):.4f}")

        # calculate the ROC AUC score

        with torch.no_grad():
            if torch.cuda.is_available():
                X_test, y_test = X_test.cuda(), y_test.cuda()
            predictions = model_same_year_median(X_test)
            auc = sklearn.metrics.roc_auc_score(y_test.cpu().numpy(), predictions.cpu().numpy())
            auc_scores.append(auc)
            print(f"Fold {fold + 1} ROC AUC: {auc:.4f}")
            print(f'All AUC scores: {auc_scores}')

        # save the model
        torch.save(model_same_year_median.state_dict(), f'results/nam_same_year_median_random_{fold}.pt')

    print(f"Mean ROC AUC: {sum(auc_scores) / len(auc_scores):.4f}")
    # write the roc auc scores, mean roc auc and standard deviation to the output file
    output_file.write("Same year above median random split\n")
    output_file.write(f"ROC AUC scores: {auc_scores}\n")
    output_file.write(f"Mean ROC AUC: {sum(auc_scores) / len(auc_scores):.4f}\n")
    output_file.write(f"Standard deviation: {torch.tensor(auc_scores).std().item()}\n")
    output_file.write("\n")

    # train and evaluate the model for same year above median grouped split
    print("Training model for same year above median grouped split")

    auc_scores = []
    # train and evaluate the model
    for fold, (train_index, test_index) in enumerate(
            cv_grouped.split(X_same_year_median, y_same_year_median, groups_same_year_median)):
        X_train, X_test = X_same_year_median.iloc[train_index], X_same_year_median.iloc[test_index]
        y_train, y_test = y_same_year_median.iloc[train_index], y_same_year_median.iloc[test_index]

        # standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

        # create dataloaders
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

        test_data = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

        # initialize the model
        model_same_year_median = Nam(len(features), hidden_layers, hidden_size, fnet_activation_function,
                                     output_activation_function, dropout)

        if torch.cuda.is_available():
            model_same_year_median = model_same_year_median.cuda()

        #calculate the weights for the loss function
        weights = torch.tensor([y_same_year_median.value_counts()[0] / y_same_year_median.value_counts()[1]],
                               dtype=torch.float32)
        # put the weights on the GPU if available
        if torch.cuda.is_available():
            weights = weights.cuda()

        # define the loss function
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        # define the optimizer
        optimizer = torch.optim.Adam(model_same_year_median.parameters(), lr=lr)

        print(f'Number of model parameters: {sum(p.numel() for p in model_same_year_median.parameters())}')
        # Training loop
        for epoch in range(epochs):
            model_same_year_median.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                optimizer.zero_grad()
                predictions = model_same_year_median(batch_X)
                loss = loss_function(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

            # Validation phase
            model_same_year_median.eval()
            val_losses = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    if torch.cuda.is_available():
                        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                    predictions = model_same_year_median(batch_X)
                    val_losses.append(loss_function(predictions, batch_y).item())

            print(f"Fold {fold + 1} Validation Loss: {sum(val_losses) / len(val_losses):.4f}")

        # calculate the ROC AUC score

        with torch.no_grad():
            if torch.cuda.is_available():
                X_test, y_test = X_test.cuda(), y_test.cuda()
            predictions = model_same_year_median(X_test)
            auc = sklearn.metrics.roc_auc_score(y_test.cpu().numpy(), predictions.cpu().numpy())
            auc_scores.append(auc)
            print(f"Fold {fold + 1} ROC AUC: {auc:.4f}")
            print(f'All AUC scores: {auc_scores}')

        # save the model
        torch.save(model_same_year_median.state_dict(), f'results/nam_same_year_median_grouped_{fold}.pt')

    print(f"Mean ROC AUC: {sum(auc_scores) / len(auc_scores):.4f}")
    # write the roc auc scores, mean roc auc and standard deviation to the output file
    output_file.write("Same year above median grouped split\n")
    output_file.write(f"ROC AUC scores: {auc_scores}\n")
    output_file.write(f"Mean ROC AUC: {sum(auc_scores) / len(auc_scores):.4f}\n")
    output_file.write(f"Standard deviation: {torch.tensor(auc_scores).std().item()}\n")
    output_file.write("\n")

    # train and evaluate the model for same year above sd random split
    print("Training model for same year above sd random split")

    auc_scores = []
    # train and evaluate the model
    for fold, (train_index, test_index) in enumerate(cv_random.split(X_same_year_above_sd, y_same_year_above_sd)):
        X_train, X_test = X_same_year_above_sd.iloc[train_index], X_same_year_above_sd.iloc[test_index]
        y_train, y_test = y_same_year_above_sd.iloc[train_index], y_same_year_above_sd.iloc[test_index]

        # standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

        # create dataloaders
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

        test_data = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

        # initialize the model
        model_same_year_sd = Nam(len(features), hidden_layers, hidden_size, fnet_activation_function,
                                 output_activation_function, dropout)

        if torch.cuda.is_available():
            model_same_year_sd = model_same_year_sd.cuda()

        #calculate the weights for the loss function
        weights = torch.tensor([y_same_year_above_sd.value_counts()[0] / y_same_year_above_sd.value_counts()[1]],
                               dtype=torch.float32)
        # put the weights on the GPU if available
        if torch.cuda.is_available():
            weights = weights.cuda()

        # define the loss function
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        # define the optimizer
        optimizer = torch.optim.Adam(model_same_year_sd.parameters(), lr=lr)

        print(f'Number of model parameters: {sum(p.numel() for p in model_same_year_sd.parameters())}')
        # Training loop
        for epoch in range(epochs):
            model_same_year_sd.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                optimizer.zero_grad()
                predictions = model_same_year_sd(batch_X)
                loss = loss_function(predictions, batch_y)
                loss.backward()

                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

            # Validation phase
            model_same_year_sd.eval()
            val_losses = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    if torch.cuda.is_available():
                        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                    predictions = model_same_year_sd(batch_X)
                    val_losses.append(loss_function(predictions, batch_y).item())

            print(f"Fold {fold + 1} Validation Loss: {sum(val_losses) / len(val_losses):.4f}")

        # calculate the ROC AUC score
        with torch.no_grad():
            if torch.cuda.is_available():
                X_test, y_test = X_test.cuda(), y_test.cuda()
            predictions = model_same_year_sd(X_test)
            auc = sklearn.metrics.roc_auc_score(y_test.cpu().numpy(), predictions.cpu().numpy())
            auc_scores.append(auc)
            print(f"Fold {fold + 1} ROC AUC: {auc:.4f}")
            print(f'All AUC scores: {auc_scores}')

        # save the model
        torch.save(model_same_year_sd.state_dict(), f'results/nam_same_year_sd_random_{fold}.pt')

    # write the roc auc scores, mean roc auc and standard deviation to the output file
    print(f"Mean ROC AUC: {sum(auc_scores) / len(auc_scores):.4f}")
    output_file.write("Same year above sd random split\n")
    output_file.write(f"ROC AUC scores: {auc_scores}\n")
    output_file.write(f"Mean ROC AUC: {sum(auc_scores) / len(auc_scores):.4f}\n")
    output_file.write(f"Standard deviation: {torch.tensor(auc_scores).std().item()}\n")
    output_file.write("\n")

    # train and evaluate the model for same year above sd grouped split
    print("Training model for same year above sd grouped split")

    auc_scores = []

    # train and evaluate the model
    for fold, (train_index, test_index) in enumerate(cv_grouped.split(X_same_year_above_sd, y_same_year_above_sd,
                                                                      groups_same_year_above_sd)):
        X_train, X_test = X_same_year_above_sd.iloc[train_index], X_same_year_above_sd.iloc[test_index]
        y_train, y_test = y_same_year_above_sd.iloc[train_index], y_same_year_above_sd.iloc[test_index]

        # standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

        # create dataloaders
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

        test_data = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

        # initialize the model
        model_same_year_sd = Nam(len(features), hidden_layers, hidden_size, fnet_activation_function,
                                 output_activation_function, dropout)

        if torch.cuda.is_available():
            model_same_year_sd = model_same_year_sd.cuda()

        #calculate the weights for the loss function
        weights = torch.tensor([y_same_year_above_sd.value_counts()[0] / y_same_year_above_sd.value_counts()[1]],
                               dtype=torch.float32)
        # put the weights on the GPU if available
        if torch.cuda.is_available():
            weights = weights.cuda()

        # define the loss function
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=weights)

        # define the optimizer
        optimizer = torch.optim.Adam(model_same_year_sd.parameters(), lr=lr)

        print(f'Number of model parameters: {sum(p.numel() for p in model_same_year_sd.parameters())}')
        # Training loop

        for epoch in range(epochs):
            model_same_year_sd.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                optimizer.zero_grad()
                predictions = model_same_year_sd(batch_X)
                loss = loss_function(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

            # Validation phase
            model_same_year_sd.eval()
            val_losses = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    if torch.cuda.is_available():
                        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                    predictions = model_same_year_sd(batch_X)
                    val_losses.append(loss_function(predictions, batch_y).item())

            print(f"Fold {fold + 1} Validation Loss: {sum(val_losses) / len(val_losses):.4f}")

        # calculate the ROC AUC score
        with torch.no_grad():
            if torch.cuda.is_available():
                X_test, y_test = X_test.cuda(), y_test.cuda()
            predictions = model_same_year_sd(X_test)
            auc = sklearn.metrics.roc_auc_score(y_test.cpu().numpy(), predictions.cpu().numpy())
            auc_scores.append(auc)
            print(f"Fold {fold + 1} ROC AUC: {auc:.4f}")
            print(f'All AUC scores: {auc_scores}')

        # save the model
        torch.save(model_same_year_sd.state_dict(), f'results/nam_same_year_sd_grouped_{fold}.pt')

    # write the roc auc scores, mean roc auc and standard deviation to the output file
    output_file.write("Same year above sd grouped split\n")
    output_file.write(f"ROC AUC scores: {auc_scores}\n")
    output_file.write(f"Mean ROC AUC: {sum(auc_scores) / len(auc_scores):.4f}\n")
    output_file.write(f"Standard deviation: {torch.tensor(auc_scores).std().item()}\n")
    output_file.write("\n")

    # train and evaluate the model for next year above median random split
    print("Training model for next year above median random split")

    auc_scores = []
    # train and evaluate the model

    for fold, (train_index, test_index) in enumerate(cv_random.split(X_next_year_median, y_next_year_median)):
        X_train, X_test = X_next_year_median.iloc[train_index], X_next_year_median.iloc[test_index]
        y_train, y_test = y_next_year_median.iloc[train_index], y_next_year_median.iloc[test_index]

        # standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

        # create dataloaders
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

        test_data = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

        # initialize the model
        model_next_year_median = Nam(len(features), hidden_layers, hidden_size, fnet_activation_function,
                                     output_activation_function, dropout)

        if torch.cuda.is_available():
            model_next_year_median = model_next_year_median.cuda()

        #calculate the weights for the loss function
        weights = torch.tensor([y_next_year_median.value_counts()[0] / y_next_year_median.value_counts()[1]],
                               dtype=torch.float32)
        # put the weights on the GPU if available
        if torch.cuda.is_available():
            weights = weights.cuda()

        # define the loss function
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        # define the optimizer
        optimizer = torch.optim.Adam(model_next_year_median.parameters(), lr=lr)

        print(f'Number of model parameters: {sum(p.numel() for p in model_next_year_median.parameters())}')
        # Training loop
        for epoch in range(epochs):
            model_next_year_median.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                optimizer.zero_grad()
                predictions = model_next_year_median(batch_X)
                loss = loss_function(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

            # Validation phase
            model_next_year_median.eval()
            val_losses = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    if torch.cuda.is_available():
                        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                    predictions = model_next_year_median(batch_X)
                    val_losses.append(loss_function(predictions, batch_y).item())

            print(f"Fold {fold + 1} Validation Loss: {sum(val_losses) / len(val_losses):.4f}")

        # calculate the ROC AUC score
        with torch.no_grad():
            if torch.cuda.is_available():
                X_test, y_test = X_test.cuda(), y_test.cuda()
            predictions = model_next_year_median(X_test)
            auc = sklearn.metrics.roc_auc_score(y_test.cpu().numpy(), predictions.cpu().numpy())
            auc_scores.append(auc)
            print(f"Fold {fold + 1} ROC AUC: {auc:.4f}")
            print(f'All AUC scores: {auc_scores}')

        # save the model
        torch.save(model_next_year_median.state_dict(), f'results/nam_next_year_median_random_{fold}.pt')

    # Report final cross-validated AUC
    print(f"Mean ROC AUC: {sum(auc_scores) / len(auc_scores):.4f}")
    # write the roc auc scores, mean roc auc and standard deviation to the output file
    output_file.write("Next year above median random split\n")
    output_file.write(f"ROC AUC scores: {auc_scores}\n")
    output_file.write(f"Mean ROC AUC: {sum(auc_scores) / len(auc_scores):.4f}\n")
    output_file.write(f"Standard deviation: {torch.tensor(auc_scores).std().item()}\n")
    output_file.write("\n")

    # train and evaluate the model for next year above median grouped split
    print("Training model for next year above median grouped split")

    auc_scores = []
    # train and evaluate the model

    for fold, (train_index, test_index) in enumerate(
            cv_grouped.split(X_next_year_median, y_next_year_median, groups_next_year_median)):
        X_train, X_test = X_next_year_median.iloc[train_index], X_next_year_median.iloc[test_index]
        y_train, y_test = y_next_year_median.iloc[train_index], y_next_year_median.iloc[test_index]

        # standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

        # create dataloaders
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

        test_data = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

        # initialize the model
        model_next_year_median = Nam(len(features), hidden_layers, hidden_size, fnet_activation_function,
                                     output_activation_function, dropout)

        if torch.cuda.is_available():
            model_next_year_median = model_next_year_median.cuda()

        #calculate the weights for the loss function
        weights = torch.tensor([y_next_year_median.value_counts()[0] / y_next_year_median.value_counts()[1]],
                               dtype=torch.float32)
        # put the weights on the GPU if available
        if torch.cuda.is_available():
            weights = weights.cuda()

        # define the loss function
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        # define the optimizer
        optimizer = torch.optim.Adam(model_next_year_median.parameters(), lr=lr)

        print(f'Number of model parameters: {sum(p.numel() for p in model_next_year_median.parameters())}')
        # Training loop
        for epoch in range(epochs):
            model_next_year_median.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                optimizer.zero_grad()
                predictions = model_next_year_median(batch_X)
                loss = loss_function(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

            # Validation phase
            model_next_year_median.eval()
            val_losses = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    if torch.cuda.is_available():
                        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                    predictions = model_next_year_median(batch_X)
                    val_losses.append(loss_function(predictions, batch_y).item())
            print(f"Fold {fold + 1} Validation Loss: {sum(val_losses) / len(val_losses):.4f}")

        # calculate the ROC AUC score
        with torch.no_grad():
            if torch.cuda.is_available():
                X_test, y_test = X_test.cuda(), y_test.cuda()
            predictions = model_next_year_median(X_test)
            auc = sklearn.metrics.roc_auc_score(y_test.cpu().numpy(), predictions.cpu().numpy())
            auc_scores.append(auc)
            print(f"Fold {fold + 1} ROC AUC: {auc:.4f}")
            print(f'All AUC scores: {auc_scores}')

        # save the model
        torch.save(model_next_year_median.state_dict(), f'results/nam_next_year_median_grouped_{fold}.pt')

    # Report final cross-validated AUC
    print(f"Mean ROC AUC: {sum(auc_scores) / len(auc_scores):.4f}")
    # write the roc auc scores, mean roc auc and standard deviation to the output file
    output_file.write("Next year above median grouped split\n")
    output_file.write(f"ROC AUC scores: {auc_scores}\n")
    output_file.write(f"Mean ROC AUC: {sum(auc_scores) / len(auc_scores):.4f}\n")
    output_file.write(f"Standard deviation: {torch.tensor(auc_scores).std().item()}\n")
    output_file.write("\n")

    # train and evaluate the model for next year above sd random split
    print("Training model for next year above sd random split")

    auc_scores = []
    # train and evaluate the model

    for fold, (train_index, test_index) in enumerate(cv_random.split(X_next_year_above_sd, y_next_year_above_sd)):
        X_train, X_test = X_next_year_above_sd.iloc[train_index], X_next_year_above_sd.iloc[test_index]
        y_train, y_test = y_next_year_above_sd.iloc[train_index], y_next_year_above_sd.iloc[test_index]

        # standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

        # create dataloaders
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

        test_data = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

        # initialize the model
        model_next_year_sd = Nam(len(features), hidden_layers, hidden_size, fnet_activation_function,
                                 output_activation_function, dropout)

        if torch.cuda.is_available():
            model_next_year_sd = model_next_year_sd.cuda()

        # calculate the weights for the loss function
        weights = torch.tensor([y_next_year_above_sd.value_counts()[0] / y_next_year_above_sd.value_counts()[1]],
                               dtype=torch.float32)
        # put the weights on the GPU if available
        if torch.cuda.is_available():
            weights = weights.cuda()

        # define the loss function
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        # define the optimizer
        optimizer = torch.optim.Adam(model_next_year_sd.parameters(), lr=lr)

        print(f'Number of model parameters: {sum(p.numel() for p in model_next_year_sd.parameters())}')
        # Training loop
        for epoch in range(epochs):
            model_next_year_sd.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                optimizer.zero_grad()
                predictions = model_next_year_sd(batch_X)
                loss = loss_function(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

            # Validation phase
            model_next_year_sd.eval()
            val_losses = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    if torch.cuda.is_available():
                        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                    predictions = model_next_year_sd(batch_X)
                    val_losses.append(loss_function(predictions, batch_y).item())

            print(f"Fold {fold + 1} Validation Loss: {sum(val_losses) / len(val_losses):.4f}")

        # calculate the ROC AUC score
        with torch.no_grad():
            if torch.cuda.is_available():
                X_test, y_test = X_test.cuda(), y_test.cuda()
            predictions = model_next_year_sd(X_test)
            auc = sklearn.metrics.roc_auc_score(y_test.cpu().numpy(), predictions.cpu().numpy())
            auc_scores.append(auc)
            print(f"Fold {fold + 1} ROC AUC: {auc:.4f}")
            print(f'All AUC scores: {auc_scores}')

        # save the model
        torch.save(model_next_year_sd.state_dict(), f'results/nam_next_year_sd_random_{fold}.pt')

    # Report final cross-validated AUC
    print(f"Mean ROC AUC: {sum(auc_scores) / len(auc_scores):.4f}")
    # write the roc auc scores, mean roc auc and standard deviation to the output file
    output_file.write("Next year above sd random split\n")
    output_file.write(f"ROC AUC scores: {auc_scores}\n")
    output_file.write(f"Mean ROC AUC: {sum(auc_scores) / len(auc_scores):.4f}\n")
    output_file.write(f"Standard deviation: {torch.tensor(auc_scores).std().item()}\n")
    output_file.write("\n")

    # train and evaluate the model for next year above sd grouped split
    print("Training model for next year above sd grouped split")

    auc_scores = []
    # train and evaluate the model

    for fold, (train_index, test_index) in enumerate(cv_grouped.split(X_next_year_above_sd, y_next_year_above_sd,
                                                                      groups_next_year_above_sd)):
        X_train, X_test = X_next_year_above_sd.iloc[train_index], X_next_year_above_sd.iloc[test_index]
        y_train, y_test = y_next_year_above_sd.iloc[train_index], y_next_year_above_sd.iloc[test_index]

        # standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert data to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

        # create dataloaders
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

        test_data = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

        # initialize the model
        model_next_year_sd = Nam(len(features), hidden_layers, hidden_size, fnet_activation_function,
                                 output_activation_function, dropout)

        if torch.cuda.is_available():
            model_next_year_sd = model_next_year_sd.cuda()

        #calculate the weights for the loss function
        weights = torch.tensor([y_next_year_above_sd.value_counts()[0] / y_next_year_above_sd.value_counts()[1]],
                               dtype=torch.float32)
        # put the weights on the GPU if available
        if torch.cuda.is_available():
            weights = weights.cuda()

        # define the loss function
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        # define the optimizer
        optimizer = torch.optim.Adam(model_next_year_sd.parameters(), lr=lr)

        print(f'Number of model parameters: {sum(p.numel() for p in model_next_year_sd.parameters())}')
        # Training loop
        for epoch in range(epochs):
            model_next_year_sd.train()
            total_loss = 0

            for batch_X, batch_y in train_loader:
                if torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                optimizer.zero_grad()
                predictions = model_next_year_sd(batch_X)
                loss = loss_function(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

            # Validation phase
            model_next_year_sd.eval()
            val_losses = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    if torch.cuda.is_available():
                        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                    predictions = model_next_year_sd(batch_X)
                    val_losses.append(loss_function(predictions, batch_y).item())

                print(f"Fold {fold + 1} Validation Loss: {sum(val_losses) / len(val_losses):.4f}")

        # calculate the ROC AUC score
        with torch.no_grad():
            if torch.cuda.is_available():
                X_test, y_test = X_test.cuda(), y_test.cuda()
            predictions = model_next_year_sd(X_test)
            try:
                auc = sklearn.metrics.roc_auc_score(y_test.cpu().numpy(), predictions.cpu().numpy())
                auc_scores.append(auc)
                print(f"Fold {fold + 1} ROC AUC: {auc:.4f}")
                print(f'All AUC scores: {auc_scores}')
            except ValueError:
                print("Only one class in test set, fold is skipped")

        # save the model
        torch.save(model_next_year_sd.state_dict(), f'results/nam_next_year_sd_grouped_{fold}.pt')

    # Report final cross-validated AUC
    print(f"Mean ROC AUC: {sum(auc_scores) / len(auc_scores):.4f}")
    # write the roc auc scores, mean roc auc and standard deviation to the output file
    output_file.write("Next year above sd grouped split\n")
    output_file.write(f"ROC AUC scores: {auc_scores}\n")
    output_file.write(f"Mean ROC AUC: {sum(auc_scores) / len(auc_scores):.4f}\n")
    output_file.write(f"Standard deviation: {torch.tensor(auc_scores).std().item()}\n")
    output_file.write("\n")

    # close the output file
    output_file.close()

    print("Training complete")

    return


if __name__ == "__main__":
    main()
