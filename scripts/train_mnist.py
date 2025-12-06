#!/usr/bin/env python3
"""
Train a small MNIST model from directory structure.
Takes separate train and test directories, each with subdirectories named 0-9, 
each containing PNG images. The test set is evaluated every epoch.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf


def validate_directory_structure(data_dir):
    """
    Validate that the directory contains subdirectories named 0-9.
    Returns True if valid, False otherwise.
    """
    data_path = Path(data_dir)
    if not data_path.exists() or not data_path.is_dir():
        print(f"Error: '{data_dir}' is not a valid directory")
        return False
    
    # Check for subdirectories named 0-9
    expected_labels = set(str(i) for i in range(10))
    found_labels = set()
    
    for item in data_path.iterdir():
        if item.is_dir():
            label = item.name
            if label in expected_labels:
                found_labels.add(label)
    
    if found_labels != expected_labels:
        missing = expected_labels - found_labels
        print(f"Error: Missing required label subdirectories: {sorted(missing)}")
        print(f"Found labels: {sorted(found_labels)}")
        return False
    
    return True


def load_images_from_directory(data_dir):
    """
    Load images from directory structure.
    Each subdirectory is named with the label (0-9).
    Each subdirectory contains PNG images that are 28x28 grayscale.
    
    Returns:
        X: numpy array of shape (n_samples, 28, 28, 1) with pixel values in [0, 1]
        y: numpy array of shape (n_samples,) with labels
    """
    data_path = Path(data_dir)
    images = []
    labels = []
    
    # Process each label subdirectory
    for label in range(10):
        label_dir = data_path / str(label)
        if not label_dir.exists():
            continue
        
        # Find all PNG files in this directory
        png_files = list(label_dir.glob("*.png")) + list(label_dir.glob("*.PNG"))
        
        print(f"Loading {len(png_files)} images from label '{label}'...")
        
        for png_file in png_files:
            try:
                # Load image
                img = Image.open(png_file)
                
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Resize to 28x28 if needed
                if img.size != (28, 28):
                    img = img.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Convert to numpy array and normalize to [0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                # Reshape to (28, 28, 1)
                img_array = img_array.reshape(28, 28, 1)
                
                images.append(img_array)
                labels.append(label)
                
            except Exception as e:
                print(f"Warning: Failed to load {png_file}: {e}")
                continue
    
    if len(images) == 0:
        raise ValueError("No images were loaded from the directory")
    
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    print(f"\nLoaded {len(images)} images total")
    print(f"Image shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    return X, y


def count_model_parameters(model):
    """Count the total number of trainable parameters in a model."""
    return sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)


def create_small_mnist_model():
    """
    Create a small MNIST model with exactly 1/8 the parameters of the original.
    
    Original model (from mnist_cnn.cpp):
    - Conv1: 28x28x1 -> 26x26x32 (3x3, 32 filters)
    - Conv2: 26x26x32 -> 24x24x32 (3x3, 32 filters)
    - MaxPool: 24x24x32 -> 12x12x32
    - Conv3: 12x12x32 -> 10x10x64 (3x3, 64 filters)
    - Conv4: 10x10x64 -> 8x8x64 (3x3, 64 filters)
    - MaxPool: 8x8x64 -> 4x4x64
    - FC1: 1024 -> 256
    - FC2: 256 -> 10
    
    Original parameters (exact):
    - Conv1: 3*3*1*32 + 32 = 320
    - Conv2: 3*3*32*32 + 32 = 9,248
    - Conv3: 3*3*32*64 + 64 = 18,496
    - Conv4: 3*3*64*64 + 64 = 36,928
    - FC1: 1024*256 + 256 = 262,400
    - FC2: 256*10 + 10 = 2,570
    - Total: 329,962 parameters
    
    Target for 1/8: 329,962 / 8 = 41,245.25 parameters
    
    Final design (2 conv layers, 1 FC layer, no strided_slice):
    - Conv1: 3*3*1*25 + 25 = 250
    - Conv2: 3*3*25*25 + 25 = 5,650
    - MaxPool: 24x24x25 -> 12x12x25
    - FC: 12*12*25 = 3600 -> 10 = 3600*10 + 10 = 36,010
    - Total: 41,910 parameters (101.6% of target, very close to 1/8)
    """
    model = tf.keras.Sequential([
        # Input: 28x28x1 grayscale images
        tf.keras.layers.Input(shape=(28, 28, 1)),
        
        # First convolutional layer
        # 28x28x1 -> 26x26x25 (3x3 kernel, 25 filters, valid padding)
        tf.keras.layers.Conv2D(
            filters=25,
            kernel_size=3,
            strides=1,
            padding='valid',
            activation='relu',
            name='conv1'
        ),
        
        # Second convolutional layer
        # 26x26x25 -> 24x24x25 (3x3 kernel, 25 filters, valid padding)
        tf.keras.layers.Conv2D(
            filters=25,
            kernel_size=3,
            strides=1,
            padding='valid',
            activation='relu',
            name='conv2'
        ),
        
        # Max pooling layer
        # 24x24x25 -> 12x12x25 (2x2 pool, stride 2)
        tf.keras.layers.MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='valid',
            name='maxpool'
        ),
        
        # Flatten for fully connected layer
        tf.keras.layers.Flatten(name='flatten'),
        
        # Single fully connected layer
        # 12*12*25 = 3600 -> 10 (output classes)
        tf.keras.layers.Dense(
            units=10,
            activation=None,  # No activation, will apply softmax in loss
            name='fc'
        ),
        
        # Softmax for classification
        tf.keras.layers.Softmax(name='softmax')
    ])
    
    return model


def train_model(train_dir, test_dir, output_path='mnist_cnn.tflite', epochs=10, batch_size=32):
    """Train a small MNIST model and save as TFLite."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Validate directory structures
    print("Validating train directory structure...")
    if not validate_directory_structure(train_dir):
        print(f"Error: Train directory '{train_dir}' has invalid structure")
        sys.exit(1)
    
    print("Validating test directory structure...")
    if not validate_directory_structure(test_dir):
        print(f"Error: Test directory '{test_dir}' has invalid structure")
        sys.exit(1)
    
    # Load images from train directory
    print("\nLoading training images...")
    X_train, y_train = load_images_from_directory(train_dir)
    
    # Load images from test directory
    print("\nLoading test images...")
    X_test, y_test = load_images_from_directory(test_dir)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape[1:]}")
    print(f"Number of classes: 10")
    
    # Create model
    print("\nCreating model...")
    model = create_small_mnist_model()
    
    # Count parameters
    total_params = count_model_parameters(model)
    original_params = 329962  # Exact count from original model
    target_params = original_params // 8
    print(f"\nModel parameters: {total_params:,}")
    print(f"Original model parameters: {original_params:,}")
    print(f"Target (1/8 of original): {target_params:,}")
    print(f"Ratio: {total_params / target_params:.2%}")
    
    # Print model summary
    print("\nModel architecture:")
    model.summary()
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
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
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
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
    print(f"  True label: {y_test[0]}")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Confidence: {output_data[0][predicted_class]:.4f}")


def main():
    """Main function to handle command-line arguments and run training."""
    parser = argparse.ArgumentParser(
        description='Train a small MNIST model from directory structure'
    )
    parser.add_argument(
        'train_dir',
        type=str,
        help='Training directory containing subdirectories named 0-9, each with PNG images'
    )
    parser.add_argument(
        'test_dir',
        type=str,
        help='Test directory containing subdirectories named 0-9, each with PNG images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='mnist_cnn.tflite',
        help='Output path for trained model (default: mnist_cnn.tflite)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    args = parser.parse_args()
    
    train_model(
        args.train_dir,
        args.test_dir,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
