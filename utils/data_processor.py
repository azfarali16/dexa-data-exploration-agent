import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List

class DataProcessor:
    """Handles data cleaning, profiling, and processing tasks"""
    
    def __init__(self, file_path: str,model=None):
        """
        Initialize the data processor with a file path.
        
        Args:
            file_path: Path to the CSV or Excel file
        """
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.df = self._load_data()
        self.original_df = self.df.copy()
        self.cleaned_basic = False
        self.cleaned_ai = False #uses AI to clean data.
        # self.model = model
        
        # Clean data automatically
        self.clean_data_basic()
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load data from file path.
        
        Returns:
            DataFrame: Loaded pandas DataFrame
        """
        if self.file_path.endswith('.csv'):
            try:
                return pd.read_csv(self.file_path)
            except UnicodeDecodeError:
                # Try with different encodings
                return pd.read_csv(self.file_path, encoding='ISO-8859-1')
        elif self.file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {self.file_path}")
    
    def clean_data_basic(self) -> None:
        """
        Performs basic cleaning data:
        - Remove duplicate rows
        - Removing Null Values
        - Handle missing values
        - Convert data types
        - Handle outliers
        """
        if self.cleaned_basic:
            return
        
        # Remove duplicate rows
        self.df = self.df.drop_duplicates()
         
        # Detect and convert data types
        for column in self.df.columns:
            # Try to convert to numeric
            if self.df[column].dtype == 'object':
                try:
                    # Check if it's a numeric column
                    pd.to_numeric(self.df[column], errors='coerce')
                    # If more than 80% of values are numeric, convert
                    if self.df[column].astype(str).str.replace('.', '', regex=False).str.isdigit().mean() > 0.8:
                        self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
                except:
                    pass
                
                # Check if it's a date column
                try:
                    # Check for date-like patterns
                    if self.df[column].astype(str).str.contains(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}').mean() > 0.8:
                        self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
                except:
                    pass
        
        
        # Removing 
        null_pct = self.df.isnull().mean()
        for col in self.df.columns:
            null_pct = self.df[col].isna().mean()

            if null_pct > 0.5 or null_pct < 0.05:
                self.df = self.df.drop(columns=[col])
            elif 0.05 < null_pct < 0.5:
                self.df = self.df[self.df[col].notna()]


        # Handle missing values
        for column in self.df.columns:
            # For numeric columns, fill with median
            if pd.api.types.is_numeric_dtype(self.df[column]):
                median_val = self.df[column].median()
                if pd.isna(median_val): # Handle case where median calculation fails (e.g., all NaNs)
                     median_val = 0 
                self.df[column] = self.df[column].fillna(median_val)
            # For categorical columns, fill with mode
            elif pd.api.types.is_object_dtype(self.df[column]):
                mode_val = self.df[column].mode()
                if not mode_val.empty:
                    self.df[column] = self.df[column].fillna(mode_val[0])
                else:
                     self.df[column] = self.df[column].fillna("Unknown") # Fallback if mode is empty
            # For datetime columns, fill with the most recent date
            elif pd.api.types.is_datetime64_dtype(self.df[column]):
                    max_date = self.df[column].max()
                    if pd.isna(max_date): # Handle case where max date is NaT
                        max_date = pd.Timestamp.now() # Fallback to current time
                    self.df[column] = self.df[column].fillna(max_date)

                    
                    
                    
        # Handle outliers for numeric columns (using IQR method)
        for column in self.df.select_dtypes(include=[np.number]).columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Check for zero IQR to avoid division by zero or unnecessary capping
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                self.df[column] = np.where(
                    self.df[column] < lower_bound,
                    lower_bound,
                    np.where(
                        self.df[column] > upper_bound,
                        upper_bound,
                        self.df[column]
                    )
                )
        
        self.cleaned_basic = True
         
        
    # def clean_data_ai(self):
    #     if self.cleaned_ai:
    #         print('yay')
    #         return
    #     if not self.model:
    #         print('Model not detected')
    #         return
         
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the dataframe.
        
        Returns:
            Dict: Summary statistics and metadata
        """
        # Basic dataset info
        row_count = len(self.df)
        column_count = len(self.df.columns)
        
        # Calculate data quality score
        missing_values = self.df.isnull().sum().sum()
        total_values = row_count * column_count if column_count > 0 else 1 # Avoid division by zero for empty df
        completeness_score = 100 - (missing_values / total_values * 100) if total_values > 0 else 100
        consistency_score = 100  # Placeholder for more complex consistency metrics
        
        data_quality_score = int((completeness_score + consistency_score) / 2)
        
        # Column summaries
        columns = []
        for column in self.df.columns:
            col_data = self.df[column]
            col_summary = {
                'name': column,
                'missing_count': int(col_data.isnull().sum()),
                'missing_percentage': round(col_data.isnull().sum() / row_count * 100, 2) if row_count > 0 else 0
            }
            
            # Type-specific stats
            if pd.api.types.is_numeric_dtype(col_data):
                col_summary['type'] = 'numeric'
                # Handle cases where stats can't be calculated (e.g., all NaNs)
                col_summary['min'] = float(col_data.min()) if pd.notna(col_data.min()) else None
                col_summary['max'] = float(col_data.max()) if pd.notna(col_data.max()) else None
                col_summary['mean'] = round(float(col_data.mean()), 2) if pd.notna(col_data.mean()) else None
                col_summary['median'] = float(col_data.median()) if pd.notna(col_data.median()) else None
                col_summary['std'] = round(float(col_data.std()), 2) if pd.notna(col_data.std()) else None
            elif pd.api.types.is_datetime64_dtype(col_data):
                col_summary['type'] = 'datetime'
                min_date = col_data.min()
                max_date = col_data.max()
                col_summary['min'] = min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else None
                col_summary['max'] = max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else None
                col_summary['range_days'] = (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else None
            else:
                col_summary['type'] = 'categorical'
                col_summary['unique_count'] = int(col_data.nunique())
                
                # Get most common values
                value_counts = col_data.value_counts()
                if not value_counts.empty:
                    most_common_val = value_counts.index[0]
                    most_common_count = int(value_counts.iloc[0])
                    col_summary['most_common'] = str(most_common_val)
                    col_summary['most_common_count'] = most_common_count
                    col_summary['most_common_percentage'] = round(most_common_count / row_count * 100, 2) if row_count > 0 else 0
                else:
                    col_summary['most_common'] = None
                    col_summary['most_common_count'] = 0
                    col_summary['most_common_percentage'] = 0.0

            columns.append(col_summary)
        
        return {
            'row_count': row_count,
            'column_count': column_count,
            'data_quality_score': data_quality_score,
            'columns': columns,
            'correlations': self._get_correlations()
        }
    
    def _get_correlations(self) -> List[Dict[str, Any]]:
        """
        Calculate correlations between numeric columns.
        
        Returns:
            List: Top correlations
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return []
        
        corr_matrix = numeric_df.corr()
        correlations = []
        
        # Get the upper triangle of the correlation matrix
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                # Only add if correlation is not NaN
                if pd.notna(corr_val):
                    correlations.append({
                        'column1': col1,
                        'column2': col2,
                        'correlation': round(float(abs(corr_val)), 3),
                        'direction': 'positive' if corr_val > 0 else ('negative' if corr_val < 0 else 'neutral')
                    })
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x['correlation'], reverse=True)
        
        return correlations # Return top 5 correlations
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the processed dataframe.
        
        Returns:
            DataFrame: Processed pandas DataFrame
        """
        return self.df
    
    def get_original_dataframe(self) -> pd.DataFrame:
        """
        Get the original unprocessed dataframe.
        
        Returns:
            DataFrame: Original pandas DataFrame
        """
        return self.original_df 
