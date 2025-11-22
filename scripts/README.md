# ML Model Training Scripts

Simple ML model for Gaussian blob detection. Minimal proof-of-concept for TFLite compiler testing.

## Setup

Install dependencies:

```bash
pip install numpy scikit-learn tensorflow
```

## Training

Train a new model and save as TFLite:

```bash
python scripts/train_model.py
```

Save to a custom path:

```bash
python scripts/train_model.py --output my_model.tflite
```

## Testing

Test a saved TFLite model:

```bash
python scripts/train_model.py --test model.tflite
```

Test with a custom path:

```bash
python scripts/train_model.py --test path/to/model.tflite
```

## Model Architecture

- Input: 2 features (x, y coordinates)
- Hidden layer: 18 neurons with ReLU activation
- Output: 3 classes (softmax classification)
- No normalization or extra layers (minimal for v1 compiler testing)

