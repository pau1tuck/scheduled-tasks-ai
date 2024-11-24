# apps/insights/services/csv_processor.py
import logging
import pandas as pd  # Import pandas for date processing
from .csv.csv_reader import load_csv
from .csv.data_validator import validate_columns
from .csv.data_cleaner import clean_data
from .csv.data_filter import filter_data
from .csv.data_overview import generate_overview

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CSVProcessor:
    def __init__(self):
        """
        Initialize the CSVProcessor.
        """
        self.df = None  # Placeholder for the loaded DataFrame

    def load(self):
        """
        Load the CSV file into a Pandas DataFrame.
        """
        logging.info("Loading CSV...")
        self.df = load_csv()

    def validate(self):
        """
        Validate that the CSV contains all required columns.
        """
        logging.info("Validating CSV columns...")
        validate_columns(self.df)

    def clean(self):
        """
        Clean the DataFrame by standardizing and formatting columns.
        """
        logging.info("Cleaning data...")
        self.df = clean_data(self.df)

    # FIXME: Encapsulate this logic in data_filter.py:
    def filter(self, start_date: str, week_number: int):
        """
        Filters the data for the current (1) or past (2) week.

        Args:
            start_date (str): Start date for the dataset (YYYY-MM-DD).
            week_number (int): Week number to filter (1 = current week, 2 = past week).

        Returns:
            pd.DataFrame: Filtered DataFrame for the specified week.
        """
        logging.info(f"Filtering data for Week {week_number}...")
        start_date = pd.to_datetime(start_date)

        if week_number == 1:  # Current Week
            week_start = start_date
            week_end = start_date + pd.Timedelta(days=6)
        elif week_number == 2:  # Past Week
            week_start = start_date - pd.Timedelta(days=7)
            week_end = start_date - pd.Timedelta(days=1)
        else:
            raise ValueError("Invalid week number. Must be 1 or 2.")

        filtered_df = self.df[
            (self.df["date"] >= week_start) & (self.df["date"] <= week_end)
        ]
        logging.info(f"Filtered Week {week_number} Data: {len(filtered_df)} rows.")
        return filtered_df

    def generate_overview(self, df, label):
        """
        Generate a statistical overview for a single DataFrame.
        """
        logging.info(f"Generating statistical overview for {label}...")
        print(f"\nStatistical Overview - {label}:")
        print(df.describe())
