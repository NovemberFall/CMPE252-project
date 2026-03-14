import argparse
import sys
import joblib
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#!/usr/bin/env python3
"""
Entry point for Breast Cancer Detection project.
Provides simple CLI to train, evaluate, and predict using a RandomForest classifier
on sklearn's breast cancer dataset. Intended as a minimal starting point.
"""





if __name__ == "__main__":
    main()