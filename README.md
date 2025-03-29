### Training Data

The training data used in our study can be found in the training_data folder.

### Instructions for experiment re-creation
To install the requirements for the scripts, run the following command:
```
pip install -r requirements.txt
```

To re-create the random forest experiments, run the following command:
``` 
python rf.py
```

To re-create the neural additive model experiments and their results, run the following command:
```
python nam.py
```
The results will be output to the results folder.

To re-create the rf results in the paper, run the following command:
```
python create_results_rf.py
```
The results will be output to the results folder.

To re-create the nam sub feature net plots, run the following command:
```
python plot_feature_dists_for_nams.py
```
The plots will be output to the results/feature_plots folder.
