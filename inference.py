import os
import torch
import numpy as np
from tqdm import tqdm
from dev.models import RawAudioCNN
from dev.loaders import LibriSpeech4SpeakerRecognition

# Loads a trained model and a dataset, and performs inference on the dataset using the model.
# Calculate the accuracy of the model on the dataset and save the predictions into a txt file.



MODEL_PATH = "model/clean_4000_96.7.tmp"
DATASET_PATH = r"C:\Adverserial\VocodedLibriSpeech\VocodedLibriSpeech\LibriSpeech\train-clean-100"

