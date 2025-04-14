import pandas as pd
import numpy as np
from datetime import datetime

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        
    def load_data(self):
        """Load data from CSV file in a single-threaded manner."""
        self.data = pd.read_csv(self.file_path)
        # Convert timestamp to datetime
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        # Set timestamp as index
        self.data = self.data.set_index('timestamp')
        return self.data
    
    def filter_by_date(self, start_date, end_date):
        """Filter data by start and end dates."""
        if self.data is None:
            self.load_data()
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        return self.data[(self.data.index >= start_date) & (self.data.index <= end_date)]
