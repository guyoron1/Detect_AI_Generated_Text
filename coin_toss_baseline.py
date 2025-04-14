import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple

class CoinTossClassifier:
    def __init__(self, random_state: int = 42):
        """Initialize the coin toss classifier with a random seed."""
        self.random_state = random_state
        np.random.seed(random_state)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make random predictions (0 or 1) for each input sample."""
        n_samples = len(X)
        return np.random.randint(0, 2, size=n_samples)

def evaluate_coin_toss_baseline(test_data: pd.DataFrame) -> Tuple[float, dict]:
    """
    Evaluate the coin toss baseline model on test data.
    
    Args:
        test_data: DataFrame with 'input' and 'label' columns
        
    Returns:
        accuracy score and classification report
    """
    # Initialize model
    model = CoinTossClassifier()
    
    # Make predictions
    y_true = test_data['label'].values
    y_pred = model.predict(test_data)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    return accuracy, report

if __name__ == "__main__":
    # Read the CSV file
    csv_path = r"C:\Users\Guy\Downloads\test_essays (1).csv"
    df = pd.read_csv(csv_path)
    
    # Create random labels for testing (0 for human, 1 for AI)
    np.random.seed(42)  # for reproducibility
    random_labels = np.random.randint(0, 2, size=len(df))
    
    # Prepare data in required format
    test_data = pd.DataFrame({
        'input': df['text'],
        'label': random_labels  # Using random labels for testing
    })
    
    # Evaluate coin toss baseline
    accuracy, report = evaluate_coin_toss_baseline(test_data)
    
    print(f"\nCoin Toss Baseline Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(f"AI Generated (Class 1) - F1: {report['1']['f1-score']:.4f}")
    print(f"Human Written (Class 0) - F1: {report['0']['f1-score']:.4f}")