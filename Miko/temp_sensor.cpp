/**
 * Temperature Sensor Implementation
 * Reads from RP2040's internal die temperature sensor
 */

#include "temp_sensor.h"
#include "hardware/adc.h"

void init_temp_sensor() {
    // Initialize ADC hardware
    adc_init();

    // Enable the internal temperature sensor
    adc_set_temp_sensor_enabled(true);

    // Select ADC channel 4 (temperature sensor)
    adc_select_input(4);
}

float read_temperature() {
    // 12-bit ADC with 3.3V reference
    const float conversion_factor = 3.3f / (1 << 12);

    // Read raw ADC value
    uint16_t raw = adc_read();

    // Convert to voltage
    float voltage = raw * conversion_factor;

    // Convert to temperature using RP2040 datasheet formula:
    // T = 27 - (ADC_voltage - 0.706) / 0.001721
    float temp_c = 27.0f - (voltage - 0.706f) / 0.001721f;

    return temp_c;
}
