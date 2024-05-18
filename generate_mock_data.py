import pandas as pd
import numpy as np

# Creating a DataFrame with mock data
data = {
    'Patient ID': [f'P{i:03d}' for i in range(1, 11)],
    'Name': ['John Doe', 'Jane Smith', 'Alice Johnson', 'Robert Brown', 'Emily Davis', 'Michael Wilson', 'Sophia Martinez', 'Daniel Anderson', 'Olivia Thomas', 'William Moore'],
    'Age': np.random.randint(20, 70, size=10),
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Diagnosis': ['Eczema', 'Acne', 'Rosacea', 'Psoriasis', 'Melanoma', 'Lupus', 'Alopecia', 'Dermatitis', 'Cellulitis', 'Vitiligo'],
    'Last Visit': pd.date_range(start='2023-01-01', periods=10, freq='M').strftime('%Y-%m-%d').tolist(),
    'Next Appointment': pd.date_range(start='2024-01-01', periods=10, freq='M').strftime('%Y-%m-%d').tolist(),
    'Notes': ['Improving', 'Needs follow-up', 'Stable', 'Under observation', 'Requires biopsy', 'Monitor closely', 'Treatment ongoing', 'Recovery expected', 'No change', 'Responding well']
}

df = pd.DataFrame(data)

# Saving the mock data to a CSV file
df.to_csv('mock_data.csv', index=False)
print("Mock data saved to mock_data.csv")
