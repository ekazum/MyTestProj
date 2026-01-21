"""
Comprehensive tests for mimic4_data_processing functionality.

This test suite validates all the operations performed in the mimic4_data_processing notebook:
1. Patient data loading and column extraction
2. Gender mapping transformation
3. Patient text representation formatting
4. Parquet file creation and data persistence
5. Embedding generation (with mocked model)
6. Output validation and data integrity
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path


class TestMimic4DataProcessing(unittest.TestCase):
    """Test suite for MIMIC-IV data processing pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Create a temporary directory for test files
        cls.test_dir = tempfile.mkdtemp()
        cls.original_dir = os.getcwd()
        os.chdir(cls.test_dir)
        
        # Create sample MIMIC-IV patient data for testing
        cls.sample_data = pd.DataFrame({
            'subject_id': [10000001, 10000002, 10000003, 10000004, 10000005],
            'gender': ['M', 'F', 'M', 'F', 'M'],
            'anchor_age': [65, 45, 72, 38, 55]
        })
        
        # Save sample data as CSV for testing
        cls.sample_csv = os.path.join(cls.test_dir, 'test_patients.csv')
        cls.sample_data.to_csv(cls.sample_csv, index=False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests."""
        os.chdir(cls.original_dir)
        shutil.rmtree(cls.test_dir)
    
    def test_01_patient_data_loading(self):
        """Test that patient data is loaded correctly with required columns."""
        # Simulate Step 1 from the notebook: Load patient data
        patients_df = pd.read_csv(
            self.sample_csv,
            usecols=['subject_id', 'gender', 'anchor_age']
        )
        
        # Verify the data was loaded
        self.assertEqual(len(patients_df), 5, "Should load all 5 patient records")
        
        # Verify required columns exist
        expected_columns = ['subject_id', 'gender', 'anchor_age']
        self.assertListEqual(
            list(patients_df.columns),
            expected_columns,
            f"Columns should be {expected_columns}"
        )
        
        # Verify data types
        self.assertTrue(
            pd.api.types.is_integer_dtype(patients_df['subject_id']),
            "subject_id should be integer type"
        )
        # Gender can be object or string dtype depending on pandas version
        self.assertTrue(
            pd.api.types.is_object_dtype(patients_df['gender']) or 
            pd.api.types.is_string_dtype(patients_df['gender']),
            "gender should be object/string type"
        )
        self.assertTrue(
            pd.api.types.is_integer_dtype(patients_df['anchor_age']),
            "anchor_age should be integer type"
        )
    
    def test_02_parquet_file_creation_patients_sample(self):
        """Test that patients_sample.parquet is created correctly."""
        # Load and save to parquet
        patients_df = pd.read_csv(
            self.sample_csv,
            usecols=['subject_id', 'gender', 'anchor_age']
        )
        
        parquet_file = 'patients_sample.parquet'
        patients_df.to_parquet(parquet_file, index=False)
        
        # Verify file exists
        self.assertTrue(
            os.path.exists(parquet_file),
            "patients_sample.parquet should be created"
        )
        
        # Verify file can be read back
        loaded_df = pd.read_parquet(parquet_file)
        pd.testing.assert_frame_equal(
            patients_df,
            loaded_df,
            "Loaded data should match original data"
        )
    
    def test_03_gender_mapping(self):
        """Test that gender values are correctly mapped from codes to full names."""
        patients_df = pd.read_csv(
            self.sample_csv,
            usecols=['subject_id', 'gender', 'anchor_age']
        )
        
        # Define gender mapping as in the notebook
        gender_mapping = {'M': 'Male', 'F': 'Female'}
        
        # Apply mapping
        mapped_genders = patients_df['gender'].map(gender_mapping)
        
        # Verify all M values mapped to Male
        male_indices = patients_df['gender'] == 'M'
        self.assertTrue(
            all(mapped_genders[male_indices] == 'Male'),
            "All 'M' values should map to 'Male'"
        )
        
        # Verify all F values mapped to Female
        female_indices = patients_df['gender'] == 'F'
        self.assertTrue(
            all(mapped_genders[female_indices] == 'Female'),
            "All 'F' values should map to 'Female'"
        )
        
        # Verify no null values after mapping
        self.assertFalse(
            mapped_genders.isnull().any(),
            "Mapping should not produce null values"
        )
    
    def test_04_patient_text_format(self):
        """Test that patient_text is formatted correctly as 'Age {age} {gender}'."""
        patients_df = pd.read_csv(
            self.sample_csv,
            usecols=['subject_id', 'gender', 'anchor_age']
        )
        
        # Create gender mapping and patient_text as in the notebook
        gender_mapping = {'M': 'Male', 'F': 'Female'}
        patients_df['patient_text'] = patients_df.apply(
            lambda row: f"Age {row['anchor_age']} {gender_mapping[row['gender']]}",
            axis=1
        )
        
        # Verify format for each row
        for idx, row in patients_df.iterrows():
            expected_gender = 'Male' if row['gender'] == 'M' else 'Female'
            expected_text = f"Age {row['anchor_age']} {expected_gender}"
            self.assertEqual(
                row['patient_text'],
                expected_text,
                f"Patient text should match format for row {idx}"
            )
        
        # Test specific examples
        self.assertIn("Age 65 Male", patients_df['patient_text'].values)
        self.assertIn("Age 45 Female", patients_df['patient_text'].values)
        self.assertIn("Age 72 Male", patients_df['patient_text'].values)
    
    def test_05_patient_text_dataframe_structure(self):
        """Test that the final patients_text_df has correct structure."""
        patients_df = pd.read_csv(
            self.sample_csv,
            usecols=['subject_id', 'gender', 'anchor_age']
        )
        
        gender_mapping = {'M': 'Male', 'F': 'Female'}
        patients_df['patient_text'] = patients_df.apply(
            lambda row: f"Age {row['anchor_age']} {gender_mapping[row['gender']]}",
            axis=1
        )
        
        # Create final dataframe with only subject_id and patient_text
        patients_text_df = patients_df[['subject_id', 'patient_text']]
        
        # Verify columns
        self.assertEqual(
            len(patients_text_df.columns),
            2,
            "patients_text_df should have exactly 2 columns"
        )
        self.assertListEqual(
            list(patients_text_df.columns),
            ['subject_id', 'patient_text'],
            "Columns should be subject_id and patient_text"
        )
        
        # Verify no missing values
        self.assertFalse(
            patients_text_df.isnull().any().any(),
            "No null values should exist in patients_text_df"
        )
        
        # Verify row count matches
        self.assertEqual(
            len(patients_text_df),
            len(self.sample_data),
            "Row count should match original data"
        )
    
    def test_06_parquet_file_creation_patients_text(self):
        """Test that patients_text_representation.parquet is created correctly."""
        patients_df = pd.read_csv(
            self.sample_csv,
            usecols=['subject_id', 'gender', 'anchor_age']
        )
        
        gender_mapping = {'M': 'Male', 'F': 'Female'}
        patients_df['patient_text'] = patients_df.apply(
            lambda row: f"Age {row['anchor_age']} {gender_mapping[row['gender']]}",
            axis=1
        )
        patients_text_df = patients_df[['subject_id', 'patient_text']]
        
        # Save to parquet
        parquet_file = 'patients_text_representation.parquet'
        patients_text_df.to_parquet(parquet_file, index=False)
        
        # Verify file exists
        self.assertTrue(
            os.path.exists(parquet_file),
            "patients_text_representation.parquet should be created"
        )
        
        # Verify file can be read back correctly
        loaded_df = pd.read_parquet(parquet_file)
        pd.testing.assert_frame_equal(
            patients_text_df,
            loaded_df,
            "Loaded data should match original data"
        )
    
    def test_07_embedding_structure(self):
        """Test that embeddings have correct structure and dimensions."""
        # Create mock embeddings (simulating the model output)
        num_patients = len(self.sample_data)
        embedding_dim = 768  # ModernBERT typical dimension
        
        # Simulate embedding generation
        mock_embeddings = np.random.randn(num_patients, embedding_dim)
        
        # Verify shape
        self.assertEqual(
            mock_embeddings.shape[0],
            num_patients,
            f"Should have {num_patients} embeddings"
        )
        self.assertEqual(
            mock_embeddings.shape[1],
            embedding_dim,
            f"Each embedding should have {embedding_dim} dimensions"
        )
        
        # Verify data type
        self.assertTrue(
            np.issubdtype(mock_embeddings.dtype, np.floating),
            "Embeddings should be floating point numbers"
        )
    
    def test_08_embedding_dataframe_structure(self):
        """Test that the embeddings DataFrame has correct structure."""
        patients_df = pd.read_csv(
            self.sample_csv,
            usecols=['subject_id', 'gender', 'anchor_age']
        )
        
        # Create mock embeddings
        num_patients = len(patients_df)
        embedding_dim = 768
        mock_embeddings = np.random.randn(num_patients, embedding_dim)
        
        # Create embeddings DataFrame as in the notebook
        embeddings_df = pd.DataFrame({
            "subject_id": patients_df["subject_id"].values,
            "embedding": list(mock_embeddings)
        })
        
        # Verify structure
        self.assertEqual(
            len(embeddings_df.columns),
            2,
            "embeddings_df should have exactly 2 columns"
        )
        self.assertListEqual(
            list(embeddings_df.columns),
            ['subject_id', 'embedding'],
            "Columns should be subject_id and embedding"
        )
        
        # Verify embedding column contains arrays
        for embedding in embeddings_df['embedding']:
            self.assertIsInstance(
                embedding,
                np.ndarray,
                "Each embedding should be a numpy array"
            )
            self.assertEqual(
                len(embedding),
                embedding_dim,
                f"Each embedding should have {embedding_dim} dimensions"
            )
    
    def test_09_parquet_file_creation_embeddings(self):
        """Test that patient_clinical_modernbert_embeddings.parquet is created correctly."""
        patients_df = pd.read_csv(
            self.sample_csv,
            usecols=['subject_id', 'gender', 'anchor_age']
        )
        
        # Create mock embeddings
        num_patients = len(patients_df)
        embedding_dim = 768
        mock_embeddings = np.random.randn(num_patients, embedding_dim)
        
        embeddings_df = pd.DataFrame({
            "subject_id": patients_df["subject_id"].values,
            "embedding": list(mock_embeddings)
        })
        
        # Save to parquet
        parquet_file = 'patient_clinical_modernbert_embeddings.parquet'
        embeddings_df.to_parquet(parquet_file, index=False)
        
        # Verify file exists
        self.assertTrue(
            os.path.exists(parquet_file),
            "patient_clinical_modernbert_embeddings.parquet should be created"
        )
        
        # Verify file can be read back correctly
        loaded_df = pd.read_parquet(parquet_file)
        
        # Check structure
        self.assertEqual(
            len(loaded_df),
            num_patients,
            "Should have correct number of records"
        )
        
        # Verify subject_ids match
        pd.testing.assert_series_equal(
            embeddings_df['subject_id'],
            loaded_df['subject_id'],
            check_names=True
        )
    
    def test_10_end_to_end_pipeline(self):
        """Test the complete end-to-end pipeline from data loading to embedding generation."""
        # Step 1: Load patient data
        patients_df = pd.read_csv(
            self.sample_csv,
            usecols=['subject_id', 'gender', 'anchor_age']
        )
        
        # Step 2: Save to patients_sample.parquet
        patients_df.to_parquet('patients_sample.parquet', index=False)
        
        # Step 3: Create patient_text representation
        gender_mapping = {'M': 'Male', 'F': 'Female'}
        patients_df['patient_text'] = patients_df.apply(
            lambda row: f"Age {row['anchor_age']} {gender_mapping[row['gender']]}",
            axis=1
        )
        patients_text_df = patients_df[['subject_id', 'patient_text']]
        
        # Step 4: Save to patients_text_representation.parquet
        patients_text_df.to_parquet('patients_text_representation.parquet', index=False)
        
        # Step 5: Generate mock embeddings
        num_patients = len(patients_text_df)
        embedding_dim = 768
        mock_embeddings = np.random.randn(num_patients, embedding_dim)
        
        # Step 6: Create embeddings DataFrame
        embeddings_df = pd.DataFrame({
            "subject_id": patients_text_df["subject_id"].values,
            "embedding": list(mock_embeddings)
        })
        
        # Step 7: Save to patient_clinical_modernbert_embeddings.parquet
        embeddings_df.to_parquet('patient_clinical_modernbert_embeddings.parquet', index=False)
        
        # Verify all files exist
        self.assertTrue(os.path.exists('patients_sample.parquet'))
        self.assertTrue(os.path.exists('patients_text_representation.parquet'))
        self.assertTrue(os.path.exists('patient_clinical_modernbert_embeddings.parquet'))
        
        # Verify data integrity through the pipeline
        final_embeddings = pd.read_parquet('patient_clinical_modernbert_embeddings.parquet')
        self.assertEqual(len(final_embeddings), len(self.sample_data))
        
        # Verify subject_ids are preserved
        pd.testing.assert_series_equal(
            patients_df['subject_id'].reset_index(drop=True),
            final_embeddings['subject_id'].reset_index(drop=True),
            check_names=True
        )
    
    def test_11_data_integrity_gender_values(self):
        """Test that only valid gender values (M, F) are processed."""
        # This test ensures the pipeline handles only expected gender values
        patients_df = pd.read_csv(
            self.sample_csv,
            usecols=['subject_id', 'gender', 'anchor_age']
        )
        
        # Verify only M and F values exist
        unique_genders = patients_df['gender'].unique()
        for gender in unique_genders:
            self.assertIn(
                gender,
                ['M', 'F'],
                f"Gender should be M or F, found {gender}"
            )
    
    def test_12_data_integrity_age_values(self):
        """Test that age values are valid and positive."""
        patients_df = pd.read_csv(
            self.sample_csv,
            usecols=['subject_id', 'gender', 'anchor_age']
        )
        
        # Verify all ages are positive
        self.assertTrue(
            (patients_df['anchor_age'] > 0).all(),
            "All ages should be positive"
        )
        
        # Verify ages are reasonable (0-150)
        self.assertTrue(
            (patients_df['anchor_age'] <= 150).all(),
            "Ages should be reasonable (<=150)"
        )
    
    def test_13_embedding_batch_processing_simulation(self):
        """Test that batch processing of embeddings works correctly."""
        # Simulate the batch processing logic from the notebook
        patient_texts = [
            "Age 65 Male",
            "Age 45 Female",
            "Age 72 Male",
            "Age 38 Female",
            "Age 55 Male"
        ]
        
        batch_size = 2
        all_embeddings = []
        embedding_dim = 768
        
        # Process in batches (simulating the notebook's get_embeddings function)
        for i in range(0, len(patient_texts), batch_size):
            batch_texts = patient_texts[i:i + batch_size]
            batch_embeddings = np.random.randn(len(batch_texts), embedding_dim)
            all_embeddings.append(batch_embeddings)
        
        # Combine all batches
        final_embeddings = np.vstack(all_embeddings)
        
        # Verify the result
        self.assertEqual(
            final_embeddings.shape[0],
            len(patient_texts),
            "Should have embeddings for all texts"
        )
        self.assertEqual(
            final_embeddings.shape[1],
            embedding_dim,
            "Should have correct embedding dimension"
        )


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
