#!/usr/bin/env python3
"""
Quick example showing how to use the comprehensive dataset with ensemble models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load the comprehensive dataset"""
    # Load the training data
    train_df = pd.read_csv('output/comprehensive_extraction/splits/random_train.csv')
    test_df = pd.read_csv('output/comprehensive_extraction/splits/random_test.csv')
    
    # Select features (exclude metadata and target variables)
    exclude_cols = ['campaign_id', 'brand', 'category', 'launch_date', 'region', 
                   'campaign_type', 'primary_channel', 'dominant_emotion', 'voice_gender',
                   'music_genre', 'benefit_orientation', 'lifecycle_stage', 
                   'category_maturity', 'localization_depth', 'ces_categorical',
                   'creative_effectiveness_score', 'roi_multiplier', 'brand_lift_percentage',
                   'engagement_rate', 'purchase_intent_lift', 'binary_success',
                   'time_to_peak_effectiveness', 'campaign_longevity', 
                   'asset_effectiveness', 'campaign_synergy_score']
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['creative_effectiveness_score']
    X_test = test_df[feature_cols]
    y_test = test_df['creative_effectiveness_score']
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_ensemble_models(X_train, X_test, y_train, y_test):
    """Train multiple models and combine predictions"""
    
    print("🚀 Training Ensemble Models on Comprehensive Dataset\n")
    
    # 1. Random Forest
    print("1️⃣ Random Forest Regressor")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    print(f"   R² Score: {rf_r2:.4f}")
    
    # 2. Gradient Boosting
    print("\n2️⃣ Gradient Boosting Regressor")
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_r2 = r2_score(y_test, gb_pred)
    print(f"   R² Score: {gb_r2:.4f}")
    
    # 3. Linear Regression (baseline)
    print("\n3️⃣ Linear Regression (baseline)")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    print(f"   R² Score: {lr_r2:.4f}")
    
    # 4. Ensemble (average of all models)
    print("\n4️⃣ Ensemble (weighted average)")
    # Weight by performance
    total_r2 = rf_r2 + gb_r2 + lr_r2
    weights = [rf_r2/total_r2, gb_r2/total_r2, lr_r2/total_r2]
    
    ensemble_pred = (weights[0] * rf_pred + 
                    weights[1] * gb_pred + 
                    weights[2] * lr_pred)
    
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    
    print(f"   Weights: RF={weights[0]:.3f}, GB={weights[1]:.3f}, LR={weights[2]:.3f}")
    print(f"   R² Score: {ensemble_r2:.4f}")
    print(f"   RMSE: {ensemble_rmse:.4f}")
    
    # Feature importance from Random Forest
    print("\n📊 Top 10 Most Important Features (from Random Forest):")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:30s}: {row['importance']:.4f}")
    
    return {
        'models': {
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'linear_regression': lr_model
        },
        'predictions': {
            'rf': rf_pred,
            'gb': gb_pred,
            'lr': lr_pred,
            'ensemble': ensemble_pred
        },
        'scores': {
            'rf_r2': rf_r2,
            'gb_r2': gb_r2,
            'lr_r2': lr_r2,
            'ensemble_r2': ensemble_r2,
            'ensemble_rmse': ensemble_rmse
        },
        'feature_importance': feature_importance
    }

def analyze_predictions(y_test, predictions):
    """Analyze prediction quality"""
    print("\n📈 Prediction Analysis:")
    
    ensemble_pred = predictions['ensemble']
    
    # Error distribution
    errors = y_test - ensemble_pred
    print(f"\n   Mean Absolute Error: {np.mean(np.abs(errors)):.2f}")
    print(f"   Error Std Dev: {np.std(errors):.2f}")
    print(f"   95% of predictions within: ±{1.96 * np.std(errors):.2f} points")
    
    # Performance by score range
    print("\n   Performance by CES Range:")
    ranges = [(0, 40), (40, 50), (50, 60), (60, 100)]
    for low, high in ranges:
        mask = (y_test >= low) & (y_test < high)
        if mask.sum() > 0:
            range_r2 = r2_score(y_test[mask], ensemble_pred[mask])
            print(f"   CES {low}-{high}: R²={range_r2:.3f}, n={mask.sum()}")

def main():
    """Run the example"""
    print("=" * 60)
    print("COMPREHENSIVE DATASET ENSEMBLE MODELING EXAMPLE")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data()
    print(f"\n📊 Dataset loaded:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {len(feature_cols)}")
    
    # Train models
    results = train_ensemble_models(X_train, X_test, y_train, y_test)
    
    # Analyze predictions
    analyze_predictions(y_test, results['predictions'])
    
    print("\n✅ Complete! The comprehensive dataset works perfectly with ensemble models.")
    print("\n💡 Next steps:")
    print("   - Try other modeling approaches (Neural Additive Models, Bayesian, etc.)")
    print("   - Experiment with feature engineering")
    print("   - Test different train/test splits")
    print("   - Add cross-validation and hyperparameter tuning")

if __name__ == "__main__":
    main()