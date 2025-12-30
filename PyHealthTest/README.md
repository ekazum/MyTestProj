# PyHealthTest - MIMIC-IV Mortality Prediction

This folder contains a Python script that demonstrates the complete PyHealth pipeline for creating a benchmark mortality prediction dataset from MIMIC-IV data.

## Overview

The script `mimic4_mortality_prediction.py` implements a comprehensive healthcare machine learning pipeline using the [PyHealth](https://github.com/sunlabuiuc/PyHealth) library, specifically for predicting in-hospital mortality using MIMIC-IV data.

## Features

The script performs the following steps:

1. **Data Loading**: Loads MIMIC-IV dataset with three key tables:
   - `diagnoses_icd`: Patient diagnosis codes
   - `procedures_icd`: Patient procedure codes  
   - `prescriptions`: Medication prescriptions

2. **Task Definition**: Applies the in-hospital mortality prediction task

3. **Preprocessing**:
   - Maps NDC (National Drug Code) to ATC Level 3 for standardized drug classification
   - Maps ICD9CM codes to ICD10CM for consistent diagnosis/procedure coding

4. **Data Splitting**: Creates train/validation/test splits (80%/10%/10%) by patient to prevent data leakage

5. **Feature Tokenization**: Inspects vocabulary statistics for conditions, procedures, and drugs

6. **Statistics Output**: Prints comprehensive dataset statistics including sample counts and mortality rates

## Requirements

### Python Libraries
```bash
pip install pyhealth
```

### Data Requirements
- MIMIC-IV dataset (requires credentialed access from PhysioNet)
- Data should be placed in a directory and the path updated in the script

## Usage

### Basic Execution
```bash
python mimic4_mortality_prediction.py
```

### Customization

Update the data path in the script:
```python
mimic4_dataset = MIMIC4Dataset(
    root="/path/to/mimiciv/",  # Update this path
    tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
    dev=True,  # Set to False for full dataset
    refresh_cache=False
)
```

## Configuration Options

- **dev=True**: Uses a small subset for testing (faster, good for development)
- **dev=False**: Uses the complete dataset (for production/final training)
- **refresh_cache=False**: Uses cached processed data if available
- **seed=42**: Random seed for reproducible data splits

## Output

The script provides detailed output including:
- Number of patients and samples
- Vocabulary sizes for different feature types
- Mortality rate statistics
- Train/validation/test split sizes
- Data distribution information

## Next Steps

After running this script, you can:

1. Create DataLoaders for batch processing:
```python
from pyhealth.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
```

2. Select and train a model:
```python
from pyhealth.models import RNN, Transformer
model = RNN(...)
```

3. Evaluate model performance on the test set

## Data Access

MIMIC-IV data can be obtained from:
- https://physionet.org/content/mimiciv/
- Requires completion of CITI training and data use agreement

## Notes

- The script uses `dev=True` by default for testing purposes
- For production use, change `dev=False` to use the full dataset
- The path `/path/to/mimiciv/` must be updated to point to your actual MIMIC-IV data directory
- All processing is done at the patient level to maintain proper data isolation

## References

- PyHealth Documentation: https://pyhealth.readthedocs.io/
- MIMIC-IV: https://physionet.org/content/mimiciv/
- Paper: Yang, C., et al. (2023). PyHealth: A Deep Learning Toolkit for Healthcare Applications
