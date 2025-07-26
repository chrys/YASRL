"""
Test for environment variable loading from .env file.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yasrl.env_loader import EnvironmentLoader, getenv


class TestEnvironmentLoader(unittest.TestCase):
    """Test environment variable loading from .env file."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instance
        EnvironmentLoader._instance = None
        EnvironmentLoader._loaded = False

    def test_env_file_loading(self):
        """Test that .env file is loaded correctly."""
        # Create a temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('TEST_VAR=test_value\n')
            f.write('TEST_INT=42\n')
            f.write('TEST_BOOL=true\n')
            temp_env_file = f.name

        try:
            # Mock find_dotenv to return our temp file
            with patch('yasrl.env_loader.find_dotenv', return_value=temp_env_file):
                loader = EnvironmentLoader()
                
                # Test that variables are loaded
                self.assertEqual(getenv('TEST_VAR'), 'test_value')
                self.assertEqual(getenv('TEST_INT'), '42')
                self.assertEqual(getenv('TEST_BOOL'), 'true')
                
        finally:
            # Clean up
            os.unlink(temp_env_file)

    def test_singleton_pattern(self):
        """Test that EnvironmentLoader follows singleton pattern."""
        loader1 = EnvironmentLoader()
        loader2 = EnvironmentLoader()
        self.assertIs(loader1, loader2)

    def test_getenv_default_values(self):
        """Test getenv with default values."""
        # Test with non-existent variable
        self.assertEqual(getenv('NON_EXISTENT_VAR', 'default'), 'default')
        self.assertEqual(getenv('NON_EXISTENT_VAR'), '')  # Default empty string

    def test_typed_getters(self):
        """Test typed environment variable getters."""
        with patch.dict(os.environ, {
            'TEST_INT': '123',
            'TEST_FLOAT': '45.67',
            'TEST_BOOL_TRUE': 'true',
            'TEST_BOOL_FALSE': 'false',
            'TEST_INVALID_INT': 'not_a_number'
        }):
            from yasrl.env_loader import get_int, get_float, get_bool
            
            # Test integer getter
            self.assertEqual(get_int('TEST_INT'), 123)
            self.assertEqual(get_int('NON_EXISTENT', 999), 999)
            self.assertEqual(get_int('TEST_INVALID_INT', 100), 100)
            
            # Test float getter
            self.assertAlmostEqual(get_float('TEST_FLOAT'), 45.67)
            self.assertEqual(get_float('NON_EXISTENT', 99.9), 99.9)
            
            # Test boolean getter
            self.assertTrue(get_bool('TEST_BOOL_TRUE'))
            self.assertFalse(get_bool('TEST_BOOL_FALSE'))
            self.assertFalse(get_bool('NON_EXISTENT'))

    def test_project_root_detection(self):
        """Test that .env file is found in project root."""
        # This test checks that the loader can find the actual .env file
        loader = EnvironmentLoader()
        
        # Check if we can load variables from the actual .env file
        # These should be the variables from the project's .env file
        google_key = getenv('GOOGLE_API_KEY')
        postgres_uri = getenv('POSTGRES_URI')
        
        # At least one of these should be set if .env is loaded
        self.assertTrue(google_key or postgres_uri, 
                       "No environment variables loaded from .env file")


if __name__ == '__main__':
    unittest.main()
