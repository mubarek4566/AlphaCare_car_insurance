import pandas as pd
import os
from path import get_path

# Define data loader class
class DataLoader:
    def __init__(self, folder_path):
        # Initialize the Folder path of the data
        self.folder_path = folder_path
    
    def load_txt_data(self):
        """
        Function to load a CSV file using the path returned by get_csv_path().
        """
        csv_path = get_path()
        try:
            data = pd.read_csv(csv_path, delimiter='|', low_memory=False)
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {csv_path}. Please check the path.")
            return None
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return csv_path