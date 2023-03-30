import pandas as pd
import sys

predictions_file = sys.argv[1]
ground_truth_file = sys.argv[2]

# Load predictions and ground truth data into pandas dataframes
predictions = pd.read_csv(predictions_file, sep='\t')
ground_truth = pd.read_csv(ground_truth_file, sep='\t')

# Extract the first column of each dataframe
pred_labels = predictions.iloc[:, 0]
truth_labels = ground_truth.iloc[:, 0]

# Compute accuracy score
accuracy = 100*(pred_labels == truth_labels).mean()

# Validate that the rest of the values for each row are the same in both dataframes
if not predictions.iloc[:, 1:].equals(ground_truth.iloc[:, 1:]):
    print('ERROR: The predicted values do not match the ground truth values.')
    sys.exit(1)

# Print the accuracy score
print(f'Accuracy: {accuracy:.2f}')
