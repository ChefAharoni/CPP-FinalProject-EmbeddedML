#!/usr/bin/env python3
"""
Automatic Temperature Data Collection
Reads from Pico's serial port and saves to CSV file automatically
"""

import serial
import serial.tools.list_ports
import argparse
import sys
from datetime import datetime
import time


def find_pico_port():
    """Automatically find the Pico's serial port"""
    ports = serial.tools.list_ports.comports()

    # Look for common Pico identifiers
    for port in ports:
        if 'usbmodem' in port.device.lower() or 'usb serial' in port.description.lower():
            return port.device

    # If not found, list available ports
    print("Could not automatically detect Pico. Available ports:")
    for port in ports:
        print(f"  {port.device}: {port.description}")

    return None


def collect_data(port, output_file, duration_minutes=None):
    """
    Collect temperature data from Pico and save to CSV

    Args:
        port: Serial port (e.g., /dev/tty.usbmodem14201)
        output_file: Output CSV filename
        duration_minutes: Optional duration in minutes (None = run until Ctrl+C)
    """
    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  Temperature Data Collection                        ║")
    print(f"╚══════════════════════════════════════════════════════╝")
    print(f"\nPort: {port}")
    print(f"Output: {output_file}")
    if duration_minutes:
        print(f"Duration: {duration_minutes} minutes")
    else:
        print(f"Duration: Until stopped (Ctrl+C)")
    print(f"\nStarting collection...\n")

    try:
        # Open serial connection
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)  # Wait for connection to stabilize

        # Open output file
        with open(output_file, 'w') as f:
            start_time = time.time()
            sample_count = 0
            header_written = False

            print("Collecting data... Press Ctrl+C to stop\n")

            while True:
                # Check duration limit if specified
                if duration_minutes and (time.time() - start_time) > (duration_minutes * 60):
                    print(f"\n✓ Reached {duration_minutes} minute duration limit")
                    break

                # Read line from serial
                if ser.in_waiting > 0:
                    try:
                        line = ser.readline().decode('utf-8').strip()

                        if line:
                            # Write to file
                            f.write(line + '\n')
                            f.flush()  # Ensure data is written immediately

                            # Check if it's the header
                            if 'temperature' in line.lower() and not header_written:
                                header_written = True
                                print(f"Header: {line}")
                                print("-" * 50)

                            # If it's a data line (numeric), count it
                            elif header_written:
                                try:
                                    temp_val = float(line)
                                    sample_count += 1

                                    # Print progress every 10 samples
                                    if sample_count % 10 == 0:
                                        elapsed = int(time.time() - start_time)
                                        print(f"[{elapsed:4d}s] Samples: {sample_count:4d} | Latest: {temp_val:.2f}°C")
                                except ValueError:
                                    # Non-numeric line, just write it
                                    if 'DATA COLLECTION' not in line:
                                        print(f"Info: {line}")

                    except UnicodeDecodeError:
                        # Skip malformed data
                        pass

        print(f"\n╔══════════════════════════════════════════════════════╗")
        print(f"║  Collection Complete!                               ║")
        print(f"╚══════════════════════════════════════════════════════╝")
        print(f"\nTotal samples: {sample_count}")
        print(f"Duration: {int(time.time() - start_time)} seconds")
        print(f"Saved to: {output_file}\n")

    except serial.SerialException as e:
        print(f"\n❌ Error: Could not open serial port: {e}")
        print(f"Make sure:")
        print(f"  1. Pico is connected")
        print(f"  2. No other program is using the serial port")
        print(f"  3. You have permission to access the port")
        sys.exit(1)

    except KeyboardInterrupt:
        elapsed = int(time.time() - start_time)
        print(f"\n\n✓ Stopped by user")
        print(f"Collected {sample_count} samples in {elapsed} seconds")
        print(f"Saved to: {output_file}\n")

    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()


def main():
    parser = argparse.ArgumentParser(
        description='Automatically collect temperature data from Pico to CSV'
    )
    parser.add_argument(
        '--port', '-p',
        type=str,
        help='Serial port (e.g., /dev/tty.usbmodem14201). Auto-detected if not specified.'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=f'temperature_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        help='Output CSV filename (default: temperature_data_TIMESTAMP.csv)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        help='Collection duration in minutes (default: run until stopped)'
    )

    args = parser.parse_args()

    # Find port if not specified
    port = args.port
    if not port:
        port = find_pico_port()
        if not port:
            print("\n❌ Could not find Pico. Please specify port with --port")
            print("Example: python collect_temperature_data.py --port /dev/tty.usbmodem14201")
            sys.exit(1)
        print(f"✓ Auto-detected Pico on port: {port}\n")

    # Collect data
    collect_data(port, args.output, args.duration)


if __name__ == '__main__':
    main()
