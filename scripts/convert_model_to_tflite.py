#!/usr/bin/env python3
"""
Convert a Keras .h5 model to TensorFlow Lite (.tflite) format.
"""

import argparse
import os
import sys
import traceback
import tensorflow as tf


def convert_h5_to_tflite(input_path, output_path, allow_custom_ops=False, optimize=False):
    """
    Convert a Keras .h5 model to TFLite format.
    
    Args:
        input_path: Path to the input .h5 Keras model file
        output_path: Path where the output .tflite file will be saved
        allow_custom_ops: Whether to allow custom operations in TFLite
        optimize: Whether to apply default optimizations
    """
    # Validate input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Check if input file has .h5 extension (warning only)
    if not input_path.endswith('.h5'):
        print(f"Warning: Input file does not have .h5 extension: {input_path}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print(f"Loading Keras model from: {input_path}")
    try:
        # Load the Keras model
        # Note: tf.keras is available in TensorFlow 2.x
        print("Loading model (this may take a moment)...")
        model = tf.keras.models.load_model(input_path, compile=False)  # type: ignore
        print("Model loaded successfully")
        
        # Print model summary
        print("\nModel architecture:")
        model.summary()
        
        # Convert to TFLite
        print("\nConverting to TFLite format...")
        try:
            # First, try direct conversion
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            if allow_custom_ops:
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS
                ]
                converter._experimental_lower_tensor_list_ops = False
            if optimize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
        except Exception as conv_error:
            print(f"\nDirect conversion failed: {str(conv_error)}")
            print("\nTrying alternative conversion method (via SavedModel)...")
            # Alternative: Save to SavedModel format first, then convert
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                saved_model_path = os.path.join(tmpdir, "saved_model")
                model.save(saved_model_path, save_format='tf')
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
                if allow_custom_ops:
                    converter.target_spec.supported_ops = [
                        tf.lite.OpsSet.TFLITE_BUILTINS,
                        tf.lite.OpsSet.SELECT_TF_OPS
                    ]
                    converter._experimental_lower_tensor_list_ops = False
                if optimize:
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
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
        print(f"Input dtype: {input_details[0]['dtype']}")
        print(f"Output shape: {output_details[0]['shape']}")
        print(f"Output dtype: {output_details[0]['dtype']}")
        
        print("\nConversion completed successfully!")
        
    except Exception as e:
        print(f"\nError during conversion: {str(e)}")
        print("\nFull error traceback:")
        traceback.print_exc()
        print("\nTips:")
        print("  - Try using --allow-custom-ops flag if the model uses custom operations")
        print("  - The model might contain operations not supported in TFLite")
        print("  - Check if the model was saved with an incompatible TensorFlow version")
        sys.exit(1)


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert a Keras .h5 model to TensorFlow Lite (.tflite) format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_model_to_tflite.py model.h5 model.tflite
  python convert_model_to_tflite.py --input path/to/model.h5 --output path/to/output.tflite
        """
    )
    parser.add_argument(
        'input',
        type=str,
        nargs='?',
        help='Path to the input .h5 Keras model file'
    )
    parser.add_argument(
        'output',
        type=str,
        nargs='?',
        help='Path where the output .tflite file will be saved'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        dest='input_path',
        help='Path to the input .h5 Keras model file (alternative to positional argument)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        dest='output_path',
        help='Path where the output .tflite file will be saved (alternative to positional argument)'
    )
    parser.add_argument(
        '--allow-custom-ops',
        action='store_true',
        help='Allow custom operations in TFLite (may be needed for some models)'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Apply default optimizations to reduce model size'
    )

    args = parser.parse_args()
    
    # Determine input and output paths
    input_path = args.input_path or args.input
    output_path = args.output_path or args.output
    
    # Validate that both paths are provided
    if not input_path:
        parser.error("Input path is required. Use positional argument or --input/-i")
    if not output_path:
        parser.error("Output path is required. Use positional argument or --output/-o")
    
    # Convert the model
    convert_h5_to_tflite(input_path, output_path, 
                        allow_custom_ops=args.allow_custom_ops,
                        optimize=args.optimize)


if __name__ == '__main__':
    main()

