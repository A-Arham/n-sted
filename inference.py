# inference.py

import torch
import numpy as np
import scipy.io
import os
from model import UNetEEG  # Make sure the model file is available

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load the Model ---
model = UNetEEG(in_channels=129).to(device)
model.load_state_dict(torch.load("unet_eeg_weights.pth", map_location=device))
model.eval()

# --- Load and Preprocess EEG Data ---
def load_and_preprocess(file_path, trial_length=374):
    """
    Loads EEG from a .mat file, standardizes it, and segments it into trials.
    Returns segmented data and standardization parameters.
    """
    mat_data = scipy.io.loadmat(file_path)
    raw_eeg = mat_data['ceeg']  # Shape: (129, T)
    
    channel_means = np.mean(raw_eeg, axis=1, keepdims=True)
    channel_stds = np.std(raw_eeg, axis=1, keepdims=True)
    standardized_eeg = (raw_eeg - channel_means) / channel_stds

    total_samples = standardized_eeg.shape[1]
    num_trials = total_samples // trial_length
    used_samples = num_trials * trial_length
    
    segmented = standardized_eeg[:, :used_samples].reshape(129, num_trials, trial_length)
    segmented_data = np.transpose(segmented, (1, 0, 2))  # Shape: (num_trials, 129, trial_length)
    
    return segmented_data, num_trials, channel_means, channel_stds

# --- Inference for a Single Trial ---
def run_inference(eeg_array: np.ndarray):
    """
    Runs model inference on a single EEG trial.
    Input: [129, T] standardized EEG.
    Output: [T] list of prediction scores.
    """
    if eeg_array.shape[0] != 129:
        raise ValueError("Expected EEG shape (129, T)")
    
    eeg_tensor = torch.tensor(eeg_array, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(eeg_tensor)
    return prediction.squeeze().cpu().numpy().tolist()

# --- Save Channel Average Across Trials ---
def save_average_channel(segmented_data, channel_means, channel_stds, channel_index=2):
    """
    De-standardizes data and saves average trace of selected channel.
    """
    de_standardized = segmented_data * channel_stds + channel_means
    channel_data = de_standardized[:, channel_index, :]  # (num_trials, trial_length)
    average_trace = np.mean(channel_data, axis=0)
    
    save_folder = 'channel_averages'
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f'channel_{channel_index}_average.mat')
    scipy.io.savemat(save_path, {f'channel_{channel_index}_average': average_trace})
    print(f"Channel {channel_index} average saved to {save_path}")

# --- Full Pipeline Entry ---
def test_pipeline(file_path, trial_length=374, channel_index_to_save=2):
    """
    Complete pipeline:
    - Load & standardize EEG.
    - Run inference on the first trial.
    - Save average trace of a selected channel.
    """
    segmented_data, num_trials, channel_means, channel_stds = load_and_preprocess(file_path, trial_length)
    
    # Example: Run inference on the first trial
    first_trial = segmented_data[0]
    predictions = run_inference(first_trial)
    
    # Save average of selected channel
    save_average_channel(segmented_data, channel_means, channel_stds, channel_index=channel_index_to_save)
    
    return predictions

