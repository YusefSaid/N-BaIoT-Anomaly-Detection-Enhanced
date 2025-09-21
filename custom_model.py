from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
from custom_data_loader import NBaloTDataLoader

class NBaloTAnomalyDetector:
    def __init__(self, data_dir='data'):
        self.data_loader = NBaloTDataLoader(data_dir)
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        
    def build_autoencoder(self, n_inputs):
        """Build autoencoder architecture"""
        # Input layer
        visible = Input(shape=(n_inputs,))
        
        # Encoder
        e = Dense(n_inputs)(visible)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        
        e = Dense(int(n_inputs * 0.75))(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        
        e = Dense(int(n_inputs * 0.50))(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        
        e = Dense(int(n_inputs * 0.33))(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)
        
        # Bottleneck
        n_bottleneck = int(n_inputs * 0.25)
        bottleneck = Dense(n_bottleneck)(e)
        
        # Decoder
        d = Dense(int(n_inputs * 0.33))(bottleneck)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        
        d = Dense(int(n_inputs * 0.50))(d)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        
        d = Dense(int(n_inputs * 0.75))(d)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        
        d = Dense(int(n_inputs))(d)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)
        
        # Output layer
        output = Dense(n_inputs, activation='linear')(d)
        
        # Create model
        model = Model(inputs=visible, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
    
    def train_device_model(self, device_type, epochs=100, batch_size=64, train_new=True):
        """Train autoencoder for a specific device"""
        print(f"\n{'='*50}")
        print(f"Training model for device: {device_type}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # Load and prepare data
        X_train, X_val, X_test, y_test = self.data_loader.prepare_autoencoder_data(device_type)
        
        # Scale data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test) if X_test is not None else None
        
        # Store scaler
        self.scalers[device_type] = scaler
        
        n_inputs = X_train_scaled.shape[1]
        print(f"Feature dimensions: {n_inputs}")
        
        # Build model
        model = self.build_autoencoder(n_inputs)
        print(f"Model architecture:")
        model.summary()
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Callbacks
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, factor=0.5, min_lr=1e-6),
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ModelCheckpoint(
                filepath=f'models/autoencoder_{device_type}',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        if train_new:
            print("Starting training...")
            history = model.fit(
                X_train_scaled, X_train_scaled,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_scaled, X_val_scaled),
                callbacks=callbacks,
                verbose=1
            )
            
            # Plot training history
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Model Loss - {device_type}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Model Loss (Log Scale) - {device_type}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'models/training_history_{device_type}.png')
            plt.show()
        else:
            # Load existing model
            model.load_weights(f'models/autoencoder_{device_type}')
            print("Loaded existing model weights")
        
        # Store model
        self.models[device_type] = model
        
        # Calculate threshold using validation data
        val_predictions = model.predict(X_val_scaled)
        val_mse = np.mean(np.power(X_val_scaled - val_predictions, 2), axis=1)
        
        threshold = val_mse.mean() + 2 * val_mse.std()  # 2 sigma threshold
        self.thresholds[device_type] = threshold
        
        print(f"\nValidation MSE Statistics:")
        print(f"  Mean: {val_mse.mean():.6f}")
        print(f"  Std:  {val_mse.std():.6f}")
        print(f"  Min:  {val_mse.min():.6f}")
        print(f"  Max:  {val_mse.max():.6f}")
        print(f"  Threshold (mean + 2*std): {threshold:.6f}")
        
        # Evaluate on test data if available
        if X_test_scaled is not None and y_test is not None:
            self.evaluate_model(device_type, X_test_scaled, y_test)
            
            # Threshold optimization
            test_predictions = model.predict(X_test_scaled)
            test_mse = np.mean(np.power(X_test_scaled - test_predictions, 2), axis=1)
            
            # Test different thresholds
            thresholds_to_test = [
                np.percentile(val_mse, 50),  # 50th percentile
                np.percentile(val_mse, 75),  # 75th percentile  
                np.percentile(val_mse, 90),  # 90th percentile
                np.percentile(val_mse, 95),  # Current
                np.percentile(val_mse, 99),  # 99th percentile
            ]
            
            print("\nThreshold Optimization Results:")
            print("=" * 50)
            for i, threshold in enumerate(thresholds_to_test):
                y_pred = (test_mse > threshold).astype(int)
                y_true = (y_test > 0).astype(int)
                
                f1 = f1_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                
                percentile = [50, 75, 90, 95, 99][i]
                print(f"{percentile}th percentile (threshold={threshold:.6f}):")
                print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        print(f"\nTraining completed in {(time.time() - start_time)/60:.2f} minutes")
        
        return model, history if train_new else None
    
    def evaluate_model(self, device_type, X_test, y_test):
        """Evaluate model performance"""
        print(f"\nEvaluating model for {device_type}...")
        
        model = self.models[device_type]
        threshold = self.thresholds[device_type]
        
        # Get predictions
        test_predictions = model.predict(X_test)
        test_mse = np.mean(np.power(X_test - test_predictions, 2), axis=1)
        
        # Convert to binary predictions (0 = normal, 1 = anomaly)
        y_pred = (test_mse > threshold).astype(int)
        y_true = (y_test > 0).astype(int)  # All our data is attacks, so all should be 1
        
        # Calculate metrics
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\nPerformance Metrics:")
        print(f"  F1 Score:   {f1:.4f}")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Additional metrics
        if cm.sum() > 0:
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            fnr = fn / (tp + fn) if (tp + fn) > 0 else 0  # False Negative Rate
            
            print(f"\nDetailed Metrics:")
            print(f"  True Positive Rate (TPR):  {tpr:.4f}")
            print(f"  True Negative Rate (TNR):  {tnr:.4f}")
            print(f"  False Positive Rate (FPR): {fpr:.4f}")
            print(f"  False Negative Rate (FNR): {fnr:.4f}")
        
        return {
            'f1': f1, 'precision': precision, 'recall': recall,
            'confusion_matrix': cm, 'threshold': threshold
        }
    
    def train_all_devices(self, epochs=100, batch_size=64):
        """Train models for all available devices"""
        available_devices = self.data_loader.get_available_devices()
        results = {}
        
        print(f"Training models for {len(available_devices)} devices...")
        
        for device in available_devices:
            try:
                model, history = self.train_device_model(device, epochs, batch_size)
                results[device] = {'model': model, 'history': history}
            except Exception as e:
                print(f"Error training model for {device}: {e}")
                results[device] = {'error': str(e)}
        
        return results
    
    def compare_devices(self):
        """Compare performance across all trained devices"""
        print(f"\n{'='*60}")
        print("DEVICE COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        available_devices = self.data_loader.get_available_devices()
        
        for device in available_devices:
            if device in self.models:
                print(f"\nDevice: {device}")
                print(f"  Threshold: {self.thresholds[device]:.6f}")
                # Add more comparison metrics here
            else:
                print(f"\nDevice: {device} - Not trained")

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = NBaloTAnomalyDetector()
    
    # Option 1: Train a single device (recommended for testing)
    device_name = 'danmini_doorbell'  # Change this to test different devices
    
    print(f"Training model for: {device_name}")
    model, history = detector.train_device_model(
        device_name, 
        epochs=50,  # Start with fewer epochs for testing
        batch_size=64,
        train_new=True
    )
    
    # Option 2: Train all devices (uncomment to use)
    # results = detector.train_all_devices(epochs=50, batch_size=64)
    # detector.compare_devices()