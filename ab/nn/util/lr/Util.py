"""
LR Scheduler Dataset Utilities

This module provides utilities for working with the LR scheduler dataset,
including model metadata, hyperparameter extraction, and evaluation utilities.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from ab.nn.util.Util import *


def get_lr_scheduler_models(dataset_path: str = None) -> List[str]:
    """
    Get list of all LR scheduler models.
    
    Args:
        dataset_path: Path to dataset root directory
        
    Returns:
        List of model names (lr_0001, lr_0002, etc.)
    """
    if dataset_path is None:
        dataset_path = "ab/nn/nn"
    
    models = []
    base_path = Path(dataset_path)
    
    for entry in sorted(base_path.glob("lr_*")):
        if entry.is_dir():
            models.append(entry.name)
    
    return models


def get_lr_scheduler_meta(model_path: str) -> Dict:
    """
    Extract LR scheduler metadata from a model directory.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Dictionary with scheduler metadata (architecture, scheduler_type, parameters)
    """
    meta_file = Path(model_path) / "model_meta.txt"
    
    if not meta_file.exists():
        return {}
    
    try:
        with open(meta_file, 'r') as f:
            content = f.read()
            try:
                return json.loads(content)
            except:
                return {"raw": content}
    except Exception as e:
        print(f"Error reading model metadata: {e}")
        return {}


def get_hyperparameters(model_path: str) -> Dict:
    """
    Extract hyperparameters from a model.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Dictionary with normalized hyperparameters [0.0, 1.0]
    """
    hp_file = Path(model_path) / "hp.txt"
    
    if not hp_file.exists():
        return {}
    
    try:
        with open(hp_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading hyperparameters: {e}")
        return {}


def get_evaluation_results(model_path: str) -> Optional[Dict]:
    """
    Get evaluation results for a model.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Dictionary with evaluation results or None if not evaluated
    """
    eval_file = Path(model_path) / "eval_info.json"
    
    if not eval_file.exists():
        return None
    
    try:
        with open(eval_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading evaluation results: {e}")
        return None


def get_model_accuracy(model_path: str) -> Optional[float]:
    """
    Get accuracy for a model.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Accuracy value or None if not evaluated
    """
    eval_info = get_evaluation_results(model_path)
    
    if eval_info and "eval_results" in eval_info:
        return eval_info["eval_results"][1]  # Second element is accuracy
    
    return None


def get_scheduler_type(meta: Dict) -> Optional[str]:
    """
    Extract scheduler type from model metadata.
    
    Args:
        meta: Model metadata dictionary
        
    Returns:
        Scheduler type (e.g., 'StepLR', 'CosineAnnealingLR')
    """
    if isinstance(meta, dict):
        return meta.get("scheduler", None)
    return None


def get_architecture_name(meta: Dict) -> Optional[str]:
    """
    Extract architecture name from model metadata.
    
    Args:
        meta: Model metadata dictionary
        
    Returns:
        Architecture name (e.g., 'ResNet50')
    """
    if isinstance(meta, dict):
        return meta.get("architecture", None)
    return None


def load_results_csv(csv_path: str) -> pd.DataFrame:
    """
    Load evaluation results from CSV file.
    
    Args:
        csv_path: Path to results CSV file
        
    Returns:
        DataFrame with model results
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error reading results CSV: {e}")
        return pd.DataFrame()


def get_best_models(results_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Get top-N models by accuracy.
    
    Args:
        results_df: DataFrame with model results
        top_n: Number of top models to return
        
    Returns:
        DataFrame with top models sorted by accuracy
    """
    if 'accuracy' in results_df.columns:
        return results_df.nlargest(top_n, 'accuracy')
    return pd.DataFrame()


def get_scheduler_statistics(results_df: pd.DataFrame) -> Dict:
    """
    Get statistics grouped by scheduler type.
    
    Args:
        results_df: DataFrame with model results
        
    Returns:
        Dictionary with per-scheduler statistics
    """
    if 'scheduler' not in results_df.columns or 'accuracy' not in results_df.columns:
        return {}
    
    stats = {}
    for scheduler in results_df['scheduler'].unique():
        scheduler_data = results_df[results_df['scheduler'] == scheduler]['accuracy']
        stats[scheduler] = {
            'mean': float(scheduler_data.mean()),
            'std': float(scheduler_data.std()),
            'min': float(scheduler_data.min()),
            'max': float(scheduler_data.max()),
            'count': int(len(scheduler_data))
        }
    
    return stats


def validate_lr_model(model_path: str) -> Tuple[bool, List[str]]:
    """
    Validate that a model has all required files for LR scheduler dataset.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    required_files = ['new_nn.py', 'hp.txt', 'model_meta.txt']
    errors = []
    
    model_path = Path(model_path)
    
    for required_file in required_files:
        file_path = model_path / required_file
        if not file_path.exists():
            errors.append(f"Missing required file: {required_file}")
    
    # Check if hp.txt has valid JSON
    hp_file = model_path / 'hp.txt'
    if hp_file.exists():
        try:
            with open(hp_file, 'r') as f:
                json.load(f)
        except:
            errors.append("Invalid JSON format in hp.txt")
    
    return len(errors) == 0, errors


# Export main functions
__all__ = [
    'get_lr_scheduler_models',
    'get_lr_scheduler_meta',
    'get_hyperparameters',
    'get_evaluation_results',
    'get_model_accuracy',
    'get_scheduler_type',
    'get_architecture_name',
    'load_results_csv',
    'get_best_models',
    'get_scheduler_statistics',
    'validate_lr_model'
]
