"""
Test simple pour vérifier que tous les imports fonctionnent
"""

import pytest
import sys
import os

def test_import_data_generation():
    """Test que le module data_generation peut être importé"""
    try:
        from src.data_generation import COSDataGeneratorSimple
        assert True, "Import réussi"
    except ImportError as e:
        pytest.fail(f"Import échoué: {e}")

def test_import_pandas():
    """Test que pandas est disponible"""
    import pandas as pd
    assert pd.__version__ is not None

def test_import_numpy():
    """Test que numpy est disponible"""
    import numpy as np
    assert np.__version__ is not None

def test_python_path():
    """Test que les chemins sont corrects"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(project_root, 'src')
    
    assert os.path.exists(src_path), f"Chemin src non trouvé: {src_path}"
    assert os.path.exists(os.path.join(src_path, 'data_generation.py')), "Fichier data_generation.py non trouvé"