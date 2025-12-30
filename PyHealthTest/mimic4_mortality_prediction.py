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
from pyhealth.tasks import in_hospital_mortality_mimic4_fn
from pyhealth.datasets import split_by_patient


def main():
    """
    Main function to create and process MIMIC-IV mortality prediction dataset.
    
    Pipeline Steps:
    1. Load MIMIC-IV data
    2. Apply mortality prediction task
    3. Preprocess codes (NDC to ATC, ICD9CM to ICD10CM)
    4. Split data into train/val/test sets
    5. Inspect feature statistics
    """
    
    print("=" * 80)
    print("MIMIC-IV Mortality Prediction Dataset Creation Pipeline")
    print("=" * 80)
    
    # =========================================================================
    # Step 1: Data Loading
    # =========================================================================
    print("\n[Step 1] Loading MIMIC-IV Dataset...")
    print("-" * 80)
    
    # Initialize MIMIC4Dataset with the specified tables
    # - diagnoses_icd: ICD diagnosis codes for patients
    # - procedures_icd: ICD procedure codes performed on patients
    # - prescriptions: Medication prescriptions (NDC codes)
    # dev=True loads a small subset for testing/development purposes
    
    mimic4_dataset = MIMIC4Dataset(
        root="/path/to/mimiciv/",
        tables=[
            "diagnoses_icd",     # Patient diagnosis codes
            "procedures_icd",    # Patient procedure codes
            "prescriptions"      # Patient medication prescriptions
        ],
        dev=True,  # Use development mode for testing with subset of data
        refresh_cache=False  # Use cached data if available to speed up loading
    )
    
    print(f"✓ Dataset loaded successfully")
    print(f"  - Number of patients: {len(mimic4_dataset.patients)}")
    print(f"  - Tables loaded: {', '.join(mimic4_dataset.tables)}")
    
    # =========================================================================
    # Step 2: Task Definition
    # =========================================================================
    print("\n[Step 2] Applying In-Hospital Mortality Prediction Task...")
    print("-" * 80)
    
    # Apply the in-hospital mortality prediction task function
    # This function:
    # - Defines the prediction target (mortality during hospital stay)
    # - Structures the data into samples suitable for machine learning
    # - Sets up appropriate time windows for feature extraction
    
    task_ds = mimic4_dataset.set_task(in_hospital_mortality_mimic4_fn)
    
    print(f"✓ Task applied successfully")
    print(f"  - Task type: In-Hospital Mortality Prediction")
    print(f"  - Number of samples: {len(task_ds)}")
    
    # =========================================================================
    # Step 3: Preprocessing - Code Mapping
    # =========================================================================
    print("\n[Step 3] Preprocessing - Mapping Medical Codes...")
    print("-" * 80)
    
    # Map NDC (National Drug Code) to ATC Level 3
    # NDC: US-specific drug codes (highly granular)
    # ATC Level 3: Anatomical Therapeutic Chemical classification (standardized)
    # This mapping reduces dimensionality and improves generalization
    
    print("  a) Mapping NDC drug codes to ATC Level 3...")
    task_ds = task_ds.stat_code_mapping(
        source_code="NDC",      # Source code system
        target_code="ATC",      # Target code system
        level=3                 # ATC level 3 provides therapeutic subgroup classification
    )
    print(f"     ✓ NDC codes mapped to ATC Level 3")
    
    # Map ICD9CM to ICD10CM
    # ICD9CM: Older diagnosis/procedure coding system
    # ICD10CM: Newer, more detailed coding system
    # This ensures consistency if data contains both code versions
    
    print("  b) Mapping ICD9CM codes to ICD10CM...")
    task_ds = task_ds.stat_code_mapping(
        source_code="ICD9CM",   # Source code system
        target_code="ICD10CM"   # Target code system
    )
    print(f"     ✓ ICD9CM codes mapped to ICD10CM")
    
    # =========================================================================
    # Step 4: Data Splitting
    # =========================================================================
    print("\n[Step 4] Splitting Data into Train/Validation/Test Sets...")
    print("-" * 80)
    
    # Split data by patient to prevent data leakage
    # - Each patient's data stays in one split (train, val, or test)
    # - Ratios: 80% train, 10% validation, 10% test
    # - seed=42 ensures reproducibility
    
    train_ds, val_ds, test_ds = split_by_patient(
        task_ds,
        ratios=[0.8, 0.1, 0.1],  # 80% train, 10% val, 10% test
        seed=42                   # Random seed for reproducibility
    )
    
    print(f"✓ Data split completed")
    print(f"  - Training set size:   {len(train_ds):>6} samples ({len(train_ds)/len(task_ds)*100:.1f}%)")
    print(f"  - Validation set size: {len(val_ds):>6} samples ({len(val_ds)/len(task_ds)*100:.1f}%)")
    print(f"  - Test set size:       {len(test_ds):>6} samples ({len(test_ds)/len(task_ds)*100:.1f}%)")
    
    # =========================================================================
    # Step 5: Feature Tokenization and Vocabulary Inspection
    # =========================================================================
    print("\n[Step 5] Feature Tokenization - Inspecting Vocabulary Statistics...")
    print("-" * 80)
    
    # The PyHealth dataset automatically builds vocabularies for each feature type
    # Vocabularies map medical codes to integer tokens for model input
    # Let's inspect the size of vocabularies for conditions and procedures
    
    print("  Vocabulary Statistics:")
    
    # Get vocabulary for conditions (diagnoses)
    # This shows how many unique diagnosis codes are in the dataset
    if hasattr(task_ds, 'input_info') and 'conditions' in task_ds.input_info:
        conditions_vocab_size = len(task_ds.input_info['conditions']['vocab'])
        print(f"    - Conditions (Diagnoses):  {conditions_vocab_size:>6} unique codes")
    else:
        print(f"    - Conditions (Diagnoses):  N/A (not in input_info)")
    
    # Get vocabulary for procedures
    # This shows how many unique procedure codes are in the dataset
    if hasattr(task_ds, 'input_info') and 'procedures' in task_ds.input_info:
        procedures_vocab_size = len(task_ds.input_info['procedures']['vocab'])
        print(f"    - Procedures:              {procedures_vocab_size:>6} unique codes")
    else:
        print(f"    - Procedures:              N/A (not in input_info)")
    
    # Get vocabulary for drugs (prescriptions)
    # This shows how many unique drug codes are in the dataset after ATC mapping
    if hasattr(task_ds, 'input_info') and 'drugs' in task_ds.input_info:
        drugs_vocab_size = len(task_ds.input_info['drugs']['vocab'])
        print(f"    - Drugs (Prescriptions):   {drugs_vocab_size:>6} unique codes")
    else:
        print(f"    - Drugs (Prescriptions):   N/A (not in input_info)")
    
    # Alternative method to inspect vocabularies if available
    print("\n  Note: PyHealth's Vocab class can be used for detailed inspection:")
    print("    - vocab.word2idx: mapping from codes to indices")
    print("    - vocab.idx2word: mapping from indices to codes")
    print("    - vocab.size(): total vocabulary size")
    
    # =========================================================================
    # Step 6: Output Final Statistics
    # =========================================================================
    print("\n[Step 6] Final Dataset Statistics...")
    print("-" * 80)
    
    # Print comprehensive statistics about the final dataset
    print(f"\n  Overall Dataset (task_ds):")
    print(f"    - Total samples:           {len(task_ds):>6}")
    print(f"    - Number of patients:      {len({sample['patient_id'] for sample in task_ds}):>6}")
    
    # Calculate label distribution (mortality rate)
    if len(task_ds) > 0:
        labels = [sample['label'] for sample in task_ds]
        mortality_count = sum(labels)
        mortality_rate = mortality_count / len(labels) * 100
        print(f"    - Mortality cases:         {mortality_count:>6} ({mortality_rate:.2f}%)")
        print(f"    - Survived cases:          {len(labels) - mortality_count:>6} ({100-mortality_rate:.2f}%)")
    
    # Split-specific statistics
    print(f"\n  Training Set:")
    print(f"    - Samples:                 {len(train_ds):>6}")
    print(f"    - Patients:                {len({sample['patient_id'] for sample in train_ds}):>6}")
    
    print(f"\n  Validation Set:")
    print(f"    - Samples:                 {len(val_ds):>6}")
    print(f"    - Patients:                {len({sample['patient_id'] for sample in val_ds}):>6}")
    
    print(f"\n  Test Set:")
    print(f"    - Samples:                 {len(test_ds):>6}")
    print(f"    - Patients:                {len({sample['patient_id'] for sample in test_ds}):>6}")
    
    # =========================================================================
    # Additional Information
    # =========================================================================
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
    
    return task_ds, train_ds, val_ds, test_ds


if __name__ == "__main__":
    """
    Entry point for the script.
    
    Note: This script requires MIMIC-IV data to be present at /path/to/mimiciv/
    The path should be updated to point to the actual MIMIC-IV data directory.
    
    MIMIC-IV data can be obtained from: https://physionet.org/content/mimiciv/
    (Requires credentialed access)
    """
    
    try:
        task_ds, train_ds, val_ds, test_ds = main()
        print("\n✓ Script executed successfully!")
        print("  Datasets are ready for model training and evaluation.")
        
    except FileNotFoundError as e:
        print("\n✗ Error: MIMIC-IV data not found!")
        print(f"  {e}")
        print("\n  Please update the 'root' parameter in MIMIC4Dataset to point")
        print("  to your MIMIC-IV data directory.")
        print("\n  Example: root='/actual/path/to/mimiciv/'")
        
    except ImportError as e:
        print("\n✗ Error: Required library not installed!")
        print(f"  {e}")
        print("\n  Please install pyhealth:")
        print("    pip install pyhealth")
        
    except Exception as e:
        print(f"\n✗ Unexpected error occurred: {type(e).__name__}")
        print(f"  {e}")
        raise
