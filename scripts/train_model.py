#!/usr/bin/env python3
"""
Simple ML model to detect Gaussian blobs.
This is a minimal proof-of-concept model for TFLite compiler testing.
"""

import argparse
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import tensorflow as tf


def generate_data(n_samples=800, n_features=2, n_classes=3, random_state=42):
    """Generate Gaussian blob data."""
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        n_features=n_features,
        random_state=random_state,
        cluster_std=1.0
    )
    return X, y


def train_model(output_path='model.tflite'):
    """Train a simple ML model and save as TFLite."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Generate Gaussian blob data
    n_samples = 800  # < 1000 as requested
    n_features = 2
    n_classes = 3
    
    print("Generating data...")
    X, y = generate_data(n_samples, n_features, n_classes)

    # Split into train and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape[1]}")
    print(f"Number of classes: {n_classes}")

    # Create a simple model
    # Input: 2 features (x, y coordinates)
    # Hidden layer: 18 neurons with ReLU
    # Output: 3 classes (softmax for classification)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(18, activation='relu', input_shape=(n_features,), name='hidden'),
        tf.keras.layers.Dense(n_classes, activation='softmax', name='output')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    print("\nModel architecture:")
    model.summary()

    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Convert to TFLite
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"\nTFLite model saved to: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")

    # Verify the TFLite model
    print("\nVerifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")

    # Test with a sample input
    test_sample = X_test[0:1]
    interpreter.set_tensor(input_details[0]['index'], test_sample.astype(np.float32))
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data[0])
    print(f"\nSample test:")
    print(f"  Input: {test_sample[0]}")
    print(f"  True label: {y_test[0]}")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Confidence: {output_data[0][predicted_class]:.4f}")


def test_model(model_path):
    """Test a saved TFLite model on generated data."""
    print(f"Loading TFLite model from: {model_path}")
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Input dtype: {input_details[0]['dtype']}")
    print(f"Output dtype: {output_details[0]['dtype']}")

    # Generate test data using the SAME parameters as training
    # This ensures the data distribution matches what the model was trained on
    print("\nGenerating test data...")
    n_samples = 800  # Same as training
    n_features = 2
    n_classes = 3
    
    # Generate the same dataset as training (same random_state)
    X, y = generate_data(n_samples, n_features, n_classes, random_state=42)
    
    # Split the same way as training to get the test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Testing on {len(X_test)} samples...")

    # Run inference on all test samples
    correct = 0
    total = len(X_test)
    
    for i in range(total):
        # Prepare input
        input_data = X_test[i:i+1].astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data[0])
        
        if predicted_class == y_test[i]:
            correct += 1

    accuracy = correct / total
    print(f"\nTest Results:")
    print(f"  Correct predictions: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Show some sample predictions
    print("\nSample predictions:")
    for i in range(min(5, total)):
        input_data = X_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data[0])
        confidence = output_data[0][predicted_class]
        
        print(f"  Sample {i+1}:")
        print(f"    Input: {X_test[i]}")
        print(f"    True label: {y_test[i]}")
        print(f"    Predicted: {predicted_class}")
        print(f"    Confidence: {confidence:.4f}")
        print(f"    {'✓' if predicted_class == y_test[i] else '✗'}")


def main():
    """Main function to handle command-line arguments and run training or testing."""
    parser = argparse.ArgumentParser(
        description='Train or test a simple ML model for Gaussian blob detection'
    )
    parser.add_argument(
        '--test',
        type=str,
        metavar='MODEL_PATH',
        help='Test mode: path to saved TFLite model file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='model.tflite',
        help='Output path for trained model (default: model.tflite)'
    )

    args = parser.parse_args()

    if args.test:
        # Test mode
        test_model(args.test)
    else:
        # Training mode (default)
        train_model(args.output)


if __name__ == '__main__':
    main()
