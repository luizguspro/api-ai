"""
Testes para HDFSAdapter
"""
import unittest
import pandas as pd
from unittest.mock import Mock, patch
import sys
sys.path.append('../../src')

from core.hdfs_adapter import HDFSAdapter

class TestHDFSAdapter(unittest.TestCase):
    
    def setUp(self):
        self.adapter = HDFSAdapter()
    
    @patch('fsspec.filesystem')
    def test_read_parquet(self, mock_fs):
        """Testa leitura de Parquet"""
        # Mock filesystem
        mock_file = Mock()
        mock_fs.return_value.open.return_value.__enter__.return_value = mock_file
        
        # Test
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        with patch('pandas.read_parquet', return_value=df):
            result = self.adapter.read_parquet('hdfs:///test.parquet')
        
        self.assertEqual(len(result), 2)
        self.assertIn('col1', result.columns)
    
    def test_write_parquet(self):
        """Testa escrita de Parquet"""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        
        with patch.object(self.adapter.fs, 'open'):
            self.adapter.write_parquet(df, 'hdfs:///test.parquet')
            self.adapter.fs.open.assert_called_once()

if __name__ == '__main__':
    unittest.main()
