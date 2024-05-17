import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import torchaudio
import torch
import torch.nn as nn
from data import ToneDataset, wave_df
from cnn.cnnResNet import ExtedndedCNNNetwork

batch_size = 128
epochs = 1
learning_rate = 0.001

def create_data_loader(train_data, batch_size):
  train_dataloader = DataLoader(train_data, batch_size=batch_size)
  return train_dataloader

def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    progress_bar = tqdm(data_loader, desc="Training", leave=False)  # Create the progress bar

    for batch_idx, (input, target) in enumerate(progress_bar):
        input, target = input.to(device), target.to(device)

        # Forward pass
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update progress bar with the current loss
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(data_loader)
    print(f"Epoch completed - Average Loss: {avg_loss}")
    return avg_loss

def train(model, data_loader, loss_fn, optimizer, device, epochs):
  for i in range(epochs):
    print(f"Epoch {i+1}")
    train_single_epoch(model, data_loader, loss_fn, optimizer, device)
    print("------------------------------")

  print("Finished training")


if __name__ == "__main__":
  sample_rate = 18000

  if torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"
  print(f"Using device {device}")

  # instantiating out dataset object and create data loader

  mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                  n_fft=1024,
                                                  hop_length=512,
                                                  n_mels=64)
  # ms = mel_spectrogram(signal)

  md = ToneDataset(wave_df, mel_spec, sample_rate, device)
  
  train_dataloader = create_data_loader(md, batch_size)

  # construct model and assign it to device
  cnn = ExtedndedCNNNetwork().to(device)
  print(cnn)

  # initialize loss function
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

  # train model
  train(cnn, train_dataloader, loss_fn, optimizer, device, epochs)

  # save model
  torch.save(cnn.state_dict(), "cnn.pth")
  print("Trained cnn saved at cnn.pth")