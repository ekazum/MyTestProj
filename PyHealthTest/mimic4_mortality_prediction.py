"""
MIMIC-IV Mortality Prediction Script using PyHealth

This script demonstrates the complete PyHealth pipeline for creating a benchmark
mortality prediction dataset from MIMIC-IV data.

Requirements:
- pyhealth library
- MIMIC-IV data in the specified path

Author: Generated for MyTestProj
Date: 2025-12-30
"""

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.mortality_prediction import MortalityPredictionMIMIC4
from pyhealth.datasets import split_by_patient


def load_mimic4_dataset(root_path, dev=True):
    """
    Load MIMIC-IV dataset with specified tables.
    
    Args:
        root_path (str): Path to MIMIC-IV data directory
        dev (bool): If True, loads a small subset for testing/development
    
    Returns:
        MIMIC4Dataset: Loaded dataset object
    """
    print("\n[Step 1] Loading MIMIC-IV Dataset...")
    print("-" * 80)
    
    # Initialize MIMIC4Dataset with the specified tables
    # dev=True loads a small subset for testing/development purposes
    config_path=".\\configs\\mimic4_ehr.yaml"

    mimic4_dataset = MIMIC4Dataset(
        ehr_root=root_path,
        ehr_tables=[
            "patients",           # Demographics
            "admissions",         # Admission/discharge info
            "diagnoses_icd",      # Diagnoses codes
            "procedures_icd",     # Procedure codes
            "prescriptions",      # Medications
            "labevents"
        ],
        dev=dev,  # Use development mode for testing with subset of data
        ehr_config_path=config_path
    )
    
    print(f"✓ Dataset loaded successfully")
    print(f"  - Tables loaded: {', '.join(mimic4_dataset.tables)}")
    
    return mimic4_dataset


def apply_mortality_prediction_task(mimic4_dataset):
    """
    Apply in-hospital mortality prediction task to the dataset.
    
    Args:
        mimic4_dataset (MIMIC4Dataset): The loaded MIMIC-IV dataset
    
    Returns:
        Dataset: Task-specific dataset with mortality labels
    """
    print("\n[Step 2] Applying In-Hospital Mortality Prediction Task...")
    print("-" * 80)
    
    # Create an instance of the mortality prediction task
    mortality_task = MortalityPredictionMIMIC4()
    
    # Apply the task to the dataset
    task_ds = mimic4_dataset.set_task(mortality_task)
    
    print(f"✓ Task applied successfully")
    print(f"  - Task type: In-Hospital Mortality Prediction")
    print(f"  - Number of samples: {len(task_ds)}")
    
    return task_ds


def split_dataset(task_ds, ratios=[0.8, 0.1, 0.1], seed=42):
    """
    Split dataset into train/validation/test sets by patient.
    
    Args:
        task_ds (Dataset): Dataset to split
        ratios (list): Split ratios [train, val, test]. Default: [0.8, 0.1, 0.1]
        seed (int): Random seed for reproducibility. Default: 42
    
    Returns:
        tuple: (train_ds, val_ds, test_ds)
    """
    print("\n[Step 3] Splitting Data into Train/Validation/Test Sets...")
    print("-" * 80)
    
    train_ds, val_ds, test_ds = split_by_patient(
        task_ds,
        ratios=ratios,
        seed=seed
    )
    
    print(f"✓ Data split completed")
    print(f"  - Training set size:   {len(train_ds):>6} samples ({len(train_ds)/len(task_ds)*100:.1f}%)")
    print(f"  - Validation set size: {len(val_ds):>6} samples ({len(val_ds)/len(task_ds)*100:.1f}%)")
    print(f"  - Test set size:       {len(test_ds):>6} samples ({len(test_ds)/len(task_ds)*100:.1f}%)")
    
    return train_ds, val_ds, test_ds


def inspect_vocabulary_statistics(task_ds):
    """
    Inspect and print vocabulary statistics for different feature types.
    
    Args:
        task_ds (Dataset): Task dataset with vocabularies
    """
    print("\n[Step 4] Feature Tokenization - Inspecting Vocabulary Statistics...")
    print("-" * 80)
    
    print("  Vocabulary Statistics:")
    
    if hasattr(task_ds, 'input_processors') and 'conditions' in task_ds.input_processors:
        conditions_vocab_size = len(task_ds.input_processors['conditions'].code_vocab)
        print(f"    - Conditions (Diagnoses):  {conditions_vocab_size:>6} unique codes")
    else:
        print(f"    - Conditions (Diagnoses):  N/A (not in input_info)")
    
    if hasattr(task_ds, 'input_processors') and 'procedures' in task_ds.input_processors:
        procedures_vocab_size = len(task_ds.input_processors['procedures'].code_vocab)
        print(f"    - Procedures:              {procedures_vocab_size:>6} unique codes")
    else:
        print(f"    - Procedures:              N/A (not in input_info)")
    
    if hasattr(task_ds, 'input_processors') and 'drugs' in task_ds.input_processors:
        drugs_vocab_size = len(task_ds.input_processors['drugs'].code_vocab)
        print(f"    - Drugs (Prescriptions):   {drugs_vocab_size:>6} unique codes")
    else:
        print(f"    - Drugs (Prescriptions):   N/A (not in input_info)")

    if hasattr(task_ds, 'input_processors') and 'labevents' in task_ds.input_processors:
        labevents_vocab_size = len(task_ds.input_processors['labevents'].code_vocab)
        print(f"    - Lab events:   {labevents_vocab_size:>6} unique codes")
    else:
        print(f"    - Lab events:   N/A (not in input_info)")

def print_dataset_statistics(task_ds, train_ds, val_ds, test_ds):
    """
    Print comprehensive statistics about the dataset and splits.
    
    Args:
        task_ds (Dataset): Complete task dataset
        train_ds (Dataset): Training split
        val_ds (Dataset): Validation split
        test_ds (Dataset): Test split
    """
    print("\n[Step 5] Final Dataset Statistics...")
    print("-" * 80)
    
    print(f"\n  Overall Dataset (task_ds):")
    print(f"    - Total samples:           {len(task_ds):>6}")
    print(f"    - Number of patients:      {len({sample['patient_id'] for sample in task_ds}):>6}")
    
    if len(task_ds) > 0:
        labels = [sample['mortality'] for sample in task_ds]
        mortality_count = sum(labels)
        mortality_rate = mortality_count / len(labels) * 100
        print(f"    - Mortality cases:         {mortality_count.item():>6} ({mortality_rate.item():.2f}%)")
        print(f"    - Survived cases:          {len(labels) - mortality_count.item():>6} ({100-mortality_rate.item():.2f}%)")
    
    print(f"\n  Training Set:")
    print(f"    - Samples:                 {len(train_ds):>6}")
    print(f"    - Patients:                {len({sample['patient_id'] for sample in train_ds}):>6}")
    
    print(f"\n  Validation Set:")
    print(f"    - Samples:                 {len(val_ds):>6}")
    print(f"    - Patients:                {len({sample['patient_id'] for sample in val_ds}):>6}")
    
    print(f"\n  Test Set:")
    print(f"    - Samples:                 {len(test_ds):>6}")
    print(f"    - Patients:                {len({sample['patient_id'] for sample in test_ds}):>6}")


def print_sample_record(train_ds):
    """
    Prints a single sample record from the training dataset.

    Args:
        train_ds (Dataset): The training dataset.
    """
    print("\n[Step 6] Inspecting a Sample Record from the Training Dataset...")
    print("-" * 80)
    if len(train_ds) > 0:
        sample = train_ds[0]
        print("  Sample Record:")
        for key, value in sample.items():
            print(f"    - {key}: {value}")
    else:
        print("  Training dataset is empty.")


def print_next_steps():
    """Print information about next steps in the pipeline."""
    print("\n" + "=" * 80)
    print("Pipeline Completed Successfully!")
    print("=" * 80)
    
    print("\n  Next Steps:")
    print("    1. Use pyhealth.data.DataLoader to create batches for model training")
    print("    2. Choose a model from pyhealth.models (e.g., RNN, Transformer)")
    print("    3. Train the model on train_ds and validate on val_ds")
    print("    4. Evaluate final performance on test_ds")
    
    print("\n  Example DataLoader Usage:")
    print("    from pyhealth.data import DataLoader")
    print("    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)")
    print("    for batch in train_loader:")
    print("        # batch contains tokenized features and labels")
    print("        pass")


def main():
    """
    Main function to create and process MIMIC-IV mortality prediction dataset.
    """
    print("=" * 80)
    print("MIMIC-IV Mortality Prediction Dataset Creation Pipeline")
    print("=" * 80)
    
    mimic4_dataset = load_mimic4_dataset(root_path="C:\\Users\\Eli\\Data\\physionet.org\\files\\mimiciv\\3.1\\", dev=True)
    
    task_ds = apply_mortality_prediction_task(mimic4_dataset)
    
    train_ds, val_ds, test_ds = split_dataset(task_ds)
    
    inspect_vocabulary_statistics(task_ds)
    
    print_dataset_statistics(task_ds, train_ds, val_ds, test_ds)
    
    print_sample_record(train_ds)
    
    print_next_steps()
    
    return task_ds, train_ds, val_ds, test_ds


if __name__ == "__main__":
    """
    Entry point for the script.
    """
    
    try:
        task_ds, train_ds, val_ds, test_ds = main()
        print("\n✓ Script executed successfully!")
        print("  Datasets are ready for model training and evaluation.")
        
    except FileNotFoundError as e:
        print("\n✗ Error: MIMIC-IV data not found!")
        print(f"  {e}")
        print("\n  Please update the 'ehr_root' parameter in MIMIC4Dataset to point")
        print("  to your MIMIC-IV data directory.")
        print("\n  Example: ehr_root='/actual/path/to/mimiciv/'")
        
    except ImportError as e:
        print("\n✗ Error: Required library not installed!")
        print(f"  {e}")
        print("\n  Please install pyhealth:")
        print("    pip install pyhealth")
        
    except Exception as e:
        print(f"\n✗ Unexpected error occurred: {type(e).__name__}")
        print(f"  {e}")
        raise
