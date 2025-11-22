/**
 * Temperature Sensor Module
 * Interface for RP2040 internal temperature sensor
 */

#ifndef TEMP_SENSOR_H
#define TEMP_SENSOR_H

/**
 * Initialize the temperature sensor (ADC channel 4)
 * Call this once during setup
 */
void init_temp_sensor();

/**
 * Read the current temperature in Celsius
 * Uses the RP2040 internal temperature sensor
 *
 * @return Temperature in degrees Celsius
 */
float read_temperature();

#endif // TEMP_SENSOR_H
