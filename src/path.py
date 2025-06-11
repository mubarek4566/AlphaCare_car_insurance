import pandas as pd 
import os 

# Get the current working directory
current_dir = os.getcwd()

# Build the relative path

def get_path():
    # Path to the folder containing txt dataset
    file_path = os.path.join(current_dir, '../Data/MachineLearningRating_v3/MachineLearningRating_v3.txt')  
    return file_path