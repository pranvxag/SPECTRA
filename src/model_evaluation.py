"""
Model Evaluation Module for Student Performance Predictor
Windows-compatible version (no emojis)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os
import joblib
from pathlib import Path

# Configure matplotlib for Windows
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

class ModelEvaluator:
    """Class for evaluating machine learning models"""
    
    def __init__(self, output_dir='outputs'):
        """
        Initialize ModelEvaluator
        
        Parameters:
        -----------
        output_dir : str
            Directory to save evaluation outputs
        """
        self.output_dir = output_dir
        self.figures_dir = f'{output_dir}/figures'
        self.reports_dir = f'{output_dir}/reports'
        
        # Create directories if they don't exist
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        print("[OK] ModelEvaluator initialized")
        print(f"      Figures will be saved to: {self.figures_dir}")
        print(f"      Reports will be saved to: {self.reports_dir}")
    
    def evaluate_classification(self, model, X_test, y_test, model_name='model', threshold=10):
        """
        Evaluate classification model
        
        Parameters:
        -----------
        model : trained model
            The trained classification model
        X_test : array-like
            Test features
        y_test : array-like
            Test targets (original grades)
        model_name : str
            Name of the model for saving files
        threshold : int
            Threshold for binary classification (pass/fail)
        
        Returns:
        --------
        dict : Classification report
        """
        print(f"\n[Evaluating] {model_name}...")
        
        # Convert continuous grades to binary (pass/fail)
        y_test_binary = (y_test >= threshold).astype(int)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test_binary, y_pred)
        
        # Generate classification report
        report = classification_report(y_test_binary, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Save classification report
        report_path = f'{self.reports_dir}/{model_name}_classification_report.csv'
        report_df.to_csv(report_path)
        print(f"   [SAVED] Classification report: {report_path}")
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, model_name)
        
        # Plot ROC curve if probabilities available
        if y_pred_proba is not None:
            self._plot_roc_curve(y_test_binary, y_pred_proba, model_name)
        
        # Plot feature importance
        self._plot_feature_importance(model, X_test, model_name)
        
        # Print summary metrics
        accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n[Performance Summary]:")
        print(f"   • Accuracy:  {accuracy:.4f}")
        print(f"   • Precision: {precision:.4f}")
        print(f"   • Recall:    {recall:.4f}")
        print(f"   • F1-Score:  {f1:.4f}")
        
        return report
    
    def evaluate_regression(self, model, X_test, y_test, model_name='model', feature_names=None):
        """
        Evaluate regression model
        
        Parameters:
        -----------
        model : trained model
            The trained regression model
        X_test : array-like
            Test features
        y_test : array-like
            Test targets
        model_name : str
            Name of the model for saving files
        feature_names : list
            Names of features
        """
        print(f"\n[Evaluating] {model_name} (Regression)...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate residuals
        residuals = y_test - y_pred
        
        # Calculate metrics
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        
        print(f"\n[Regression Metrics]:")
        print(f"   • RMSE: {rmse:.4f}")
        print(f"   • MAE:  {mae:.4f}")
        print(f"   • R²:   {r2:.4f}")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Residual': residuals
        })
        pred_path = f'{self.reports_dir}/{model_name}_predictions.csv'
        predictions_df.to_csv(pred_path, index=False)
        print(f"   [SAVED] Predictions: {pred_path}")
        
        # Plot actual vs predicted
        self._plot_actual_vs_predicted(y_test, y_pred, model_name)
        
        # Plot residuals
        self._plot_residuals(y_test, y_pred, residuals, model_name)
        
        # Plot feature importance
        self._plot_feature_importance(model, X_test, model_name, feature_names)
        
        # Identify at-risk students
        self._identify_at_risk_students(y_test, y_pred, model_name)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions_df
        }
    
    def _plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        
        # Create heatmap manually
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.colorbar()
        
        # Add labels
        classes = ['At-Risk', 'Pass']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save figure
        save_path = f'{self.figures_dir}/{model_name}_confusion_matrix.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"   [SAVED] Confusion matrix: {save_path}")
    
    def _plot_roc_curve(self, y_true, y_score, model_name):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        save_path = f'{self.figures_dir}/{model_name}_roc_curve.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"   [SAVED] ROC curve: {save_path}")
    
    def _plot_feature_importance(self, model, X, model_name, feature_names=None):
        """Plot feature importance"""
        # Get feature names
        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = X.columns
            else:
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Extract feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
            title = 'Feature Importance'
        elif hasattr(model, 'coef_'):
            # Linear models
            if len(model.coef_.shape) > 1:
                importance = np.mean(np.abs(model.coef_), axis=0)
            else:
                importance = np.abs(model.coef_)
            title = 'Feature Importance (|Coefficient|)'
        else:
            print("   [WARNING] Feature importance not available for this model")
            return
        
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        
        # Plot top 10 features (or all if less than 10)
        n_features = min(10, len(feature_names))
        plt.figure(figsize=(10, 6))
        
        plt.barh(range(n_features), importance[indices[:n_features]], align='center')
        plt.yticks(range(n_features), [feature_names[i] for i in indices[:n_features]])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'{title} - {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save figure
        save_path = f'{self.figures_dir}/{model_name}_feature_importance.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"   [SAVED] Feature importance plot: {save_path}")
        
        # Save importance values to CSV
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        csv_path = f'{self.reports_dir}/{model_name}_feature_importance.csv'
        importance_df.to_csv(csv_path, index=False)
        print(f"   [SAVED] Feature importance data: {csv_path}")
    
    def _plot_actual_vs_predicted(self, y_true, y_pred, model_name):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(8, 6))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, c='blue', edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Grade', fontsize=12)
        plt.ylabel('Predicted Grade', fontsize=12)
        plt.title(f'Actual vs Predicted Grades - {model_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        save_path = f'{self.figures_dir}/{model_name}_actual_vs_predicted.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"   [SAVED] Actual vs predicted plot: {save_path}")
    
    def _plot_residuals(self, y_true, y_pred, residuals, model_name):
        """Plot residual analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, c='purple', edgecolors='black', linewidth=0.5)
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Grade', fontsize=11)
        axes[0].set_ylabel('Residuals', fontsize=11)
        axes[0].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residuals', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Residual Analysis - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        save_path = f'{self.figures_dir}/{model_name}_residuals.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"   [SAVED] Residual analysis plot: {save_path}")
    
    def _identify_at_risk_students(self, y_true, y_pred, model_name, threshold=10):
        """Identify at-risk students based on predictions"""
        at_risk_pred = y_pred < threshold
        at_risk_actual = y_true < threshold
        
        # Create summary dataframe
        risk_summary = pd.DataFrame({
            'Actual_Grade': y_true,
            'Predicted_Grade': y_pred,
            'Actual_Risk': at_risk_actual,
            'Predicted_Risk': at_risk_pred,
            'Correctly_Identified': at_risk_actual == at_risk_pred
        })
        
        # Calculate statistics
        n_at_risk = at_risk_actual.sum()
        n_predicted_risk = at_risk_pred.sum()
        correctly_identified = ((at_risk_actual == True) & (at_risk_pred == True)).sum()
        
        if n_at_risk > 0:
            recall_risk = correctly_identified / n_at_risk
        else:
            recall_risk = 0
        
        print(f"\n[At-Risk Student Identification]:")
        print(f"   • Actual at-risk students: {n_at_risk}")
        print(f"   • Predicted at-risk students: {n_predicted_risk}")
        print(f"   • Correctly identified: {correctly_identified}")
        print(f"   • Recall for at-risk: {recall_risk:.2%}")
        
        # Save to CSV
        csv_path = f'{self.reports_dir}/{model_name}_at_risk_identification.csv'
        risk_summary.to_csv(csv_path, index=False)
        print(f"   [SAVED] At-risk identification: {csv_path}")
        
        return risk_summary


# Example usage
if __name__ == "__main__":
    print("[TEST] Testing ModelEvaluator...")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    X_test = np.random.randn(100, 5)
    y_test = np.random.randint(0, 20, 100)
    
    # Create a dummy model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Train on random data (just for testing)
    X_train = np.random.randn(200, 5)
    y_train = np.random.randint(0, 2, 200)
    model.fit(X_train, y_train)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Test evaluation
    evaluator.evaluate_classification(model, X_test, y_test, model_name='test_model')
    
    print("\n" + "=" * 50)
    print("[COMPLETE] Test finished successfully!")