# Temperature Data Collection Guide

## Automatic CSV Data Collection

This guide shows you how to automatically save temperature data from your Pico to a CSV file without losing any data.

## Quick Start

### Method 1: Python Script (Recommended - Works on all platforms)

**1. Make sure DATA_COLLECTION_MODE is enabled in Miko.cpp:**
```cpp
#define DATA_COLLECTION_MODE true  // Line 19 in Miko.cpp
```

**2. Build and flash to your Pico**

**3. Run the collection script:**
```bash
cd scripts
python3 collect_temperature_data.py
```

The script will:
- ✅ Auto-detect your Pico's serial port
- ✅ Create a timestamped CSV file (e.g., `temperature_data_20250122_143022.csv`)
- ✅ Continuously save ALL data (no data loss!)
- ✅ Show progress as it collects
- ✅ Stop when you press Ctrl+C

**Optional arguments:**
```bash
# Specify custom output filename
python3 collect_temperature_data.py --output my_data.csv

# Auto-stop after 5 minutes
python3 collect_temperature_data.py --duration 5

# Manually specify serial port
python3 collect_temperature_data.py --port /dev/tty.usbmodem14201
```

### Method 2: Simple Command Line (macOS/Linux)

```bash
# Find your Pico's port
ls /dev/tty.usbmodem*

# Stream to CSV file
cat /dev/tty.usbmodem14201 > temperature_data.csv
```

Press Ctrl+C to stop.

## Data Collection Workflow

### Step 1: Collect "Normal" Temperature Data

1. Run the collection script:
   ```bash
   python3 collect_temperature_data.py --output normal_data.csv --duration 2
   ```

2. **Let the Pico sit idle** for 2 minutes (don't touch it!)

3. The script will automatically stop and save the file

### Step 2: Collect "Touched" Temperature Data

1. Run the collection script:
   ```bash
   python3 collect_temperature_data.py --output touched_data.csv --duration 3
   ```

2. **Follow the pattern:**
   - Wait 20 seconds (idle)
   - Touch the RP2040 chip for 10 seconds
   - Wait 20 seconds (idle)
   - Touch again for 10 seconds
   - Repeat 2-3 more times

3. The script will automatically stop after 3 minutes

### Step 3: Combine and Label the Data

Use the provided training script (next step) to combine both CSV files and add labels.

## Expected Output

Your CSV file will look like:
```
temperature
27.34
27.35
27.36
27.89
28.45
...
```

## Troubleshooting

**"Could not find Pico"**
- Check USB connection
- Try manually specifying the port: `python3 collect_temperature_data.py --port /dev/tty.usbmodem14201`

**"Permission denied"**
- On Linux, you may need: `sudo usermod -a -G dialout $USER` (then log out and back in)
- Or run with sudo: `sudo python3 collect_temperature_data.py`

**"Port already in use"**
- Close any serial monitors (Arduino IDE, screen, minicom, etc.)
- Close VS Code serial monitor if open

## Next Steps

After collecting data:
1. Train your model using `train_temperature_model.py` (coming next!)
2. Extract the trained weights
3. Replace placeholder weights in `temp_model_weights.h`
4. Set `DATA_COLLECTION_MODE false` in Miko.cpp
5. Rebuild and enjoy real finger detection!
