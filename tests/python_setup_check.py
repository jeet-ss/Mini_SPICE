import numpy as np
import torch
import pandas as pd
import librosa
import argparse
import scipy
import os
from data_files.dataloader import MedleyDBLoader, MDBMelodySynthLoader, MIR1KLoader
# extra imports
from data_files.dataset import CQT_Dataset
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.model import Spice_model

print("Imports Completed")
print("Conda: ", torch.cuda.is_available())
