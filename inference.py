import torch
import torchaudio
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
from torch.utils.data import DataLoader
import torchaudio
import torch
from data import ToneDataset
from cnn.cnnResNet import ExtedndedCNNNetwork

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device {device}")

class_mapping = ["a", "a#", "b", "c", "c#", "d", "d#", "e", "f", "f#", "g", "g#"]

def predict(model, inputs, targets, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
        predicted_indices = predictions.argmax(dim=1)
        predicted_labels = [class_mapping[idx.item()] for idx in predicted_indices]
        expected_labels = [class_mapping[target.item()] for target in targets]
    return predicted_labels, expected_labels

# Define the list of musical notes (shrutis)
shrutis = ["a", "a#", "b", "c", "c#", "d", "d#", "e", "f", "f#", "g", "g#"]
num_shrutis = len(shrutis)

# Define the desired length of the dataset
length = 1000

# Calculate the number of samples for each shruti to achieve balanced distribution
samples_per_shruti = length // num_shrutis
remaining_samples = length % num_shrutis

# Generate shrutis for the dataset with balanced distribution
gen_shrutis = np.array([shruti for shruti in shrutis for _ in range(samples_per_shruti)])
remaining_shrutis = np.random.choice(shrutis, size=remaining_samples, replace=False)
gen_shrutis = np.concatenate((gen_shrutis, remaining_shrutis))

# Shuffle the generated shrutis to randomize their order
np.random.shuffle(gen_shrutis)

# Generate class labels based on the index of shrutis
classes = np.array([shrutis.index(shruti) for shruti in gen_shrutis])

# Create the data dictionary
data_dict_2 = {
    "shruti": gen_shrutis,
    "amplitude": np.random.uniform(0.5, 0.6, size=length),
    "octave": np.array([np.random.randint(3, 4) for _ in range(length)]),
    "attack": np.random.uniform(0.1, 0.4, size=length),
    "decay": np.random.uniform(0.1, 0.4, size=length),
    "noise": np.random.uniform(0.001, 0.01, size=length),
    "class": classes
}

wave_df2 = pd.DataFrame(data_dict_2)
sample_rate = 18000

if __name__ == "__main__":
    # Load the model
    cnn = ExtedndedCNNNetwork()
    state_dict = torch.load("cnn.pth")
    cnn.load_state_dict(state_dict)

    # Load tone dataset
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    md = ToneDataset(wave_df2, mel_spec, sample_rate, device)

    # Initialize lists to store predictions and targets
    all_predictions = []
    all_targets = []

    batch_size = 32  # Define the batch size

    # Iterate over samples in batches
    for start_idx in range(0, len(md), batch_size):
        end_idx = min(start_idx + batch_size, len(md))
        inputs = []
        targets = []
        # Create batch
        for idx in range(start_idx, end_idx):
            input, target = md[idx]
            inputs.append(input.unsqueeze(0))  # Add batch dimension
            targets.append(target)

        inputs = torch.cat(inputs).to(device)
        targets = torch.tensor(targets).to(device)

        # Make inference for the current batch
        predicted_batch, expected_batch = predict(cnn, inputs, targets, class_mapping)
        all_predictions.extend(predicted_batch)
        all_targets.extend(expected_batch)

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    # Print metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("Confusion Matrix:")
    print(conf_matrix)