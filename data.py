import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torchaudio
from tones import SINE_WAVE
from tones.mixer import Mixer
import soundfile as sf


# Define the list of musical notes (shrutis)
shrutis = ["a", "a#", "b", "c", "c#", "d", "d#", "e", "f", "f#", "g", "g#"]
num_shrutis = len(shrutis)

# Define the desired length of the dataset
length = 30000

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
data_dict = {
    "shruti": gen_shrutis,
    "amplitude": np.random.uniform(0.5, 0.6, size=length),
    "octave": np.array([np.random.randint(3, 4) for _ in range(length)]),
    "attack": np.random.uniform(0.1, 0.4, size=length),
    "decay": np.random.uniform(0.1, 0.4, size=length),
    "noise": np.random.uniform(0.001, 0.01, size=length),
    "class": classes
}

wave_df = pd.DataFrame(data_dict)

class InMemoryMixer(Mixer):
    def get_audio_data(self):
        # Save the audio to a buffer instead of a file
        from io import BytesIO
        buffer = BytesIO()
        self.write_wav(buffer)
        buffer.seek(0)
        
        # Read the audio data from the buffer
        audio_data, sr = sf.read(buffer)
        return audio_data, sr
    
class ToneDataset(Dataset):

  def __init__(self, dataframe, transformation, target_sample_rate, device):
    self.dataframe = dataframe
    self.device = device
    self.transformation = transformation.to(self.device)
    self.target_sample_rate = target_sample_rate

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, index):
    # Create an instance of your InMemoryMixer class
    entry = self.dataframe.iloc[index]
    mixer = InMemoryMixer(self.target_sample_rate, entry["amplitude"])
    mixer.create_track(0, SINE_WAVE)  # sine, saw, square wave
    mixer.add_note(0, note=entry["shruti"], octave=entry["octave"], duration=1.0, attack=entry["attack"], decay=entry["decay"])  # octave range 3-4

    # Get the audio data directly from the mixer
    wav, sr = mixer.get_audio_data()

    # Add noise to the audio
    noise = np.random.random_sample(len(wav)) * entry["noise"]
    noisy_wav = wav + noise

    # Convert to Torch Tensor
    signal = torch.Tensor(noisy_wav).unsqueeze(dim=0)

    signal = signal.to(self.device)
    signal = self.transformation(signal)

    label = entry["class"]
    return signal, label

if __name__ == "__main__":
  sample_rate = 18000

  if torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"
  print(f"Using device {device}")

  mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                  n_fft=1024,
                                                  hop_length=512,
                                                  n_mels=64)
  # ms = mel_spectrogram(signal)

  md = ToneDataset(wave_df, mel_spec, sample_rate, device)

  print(md[2][0].shape)