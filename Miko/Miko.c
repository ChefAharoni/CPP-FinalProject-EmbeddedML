#include <stdio.h>
#include "pico/stdlib.h"


int main()
{
    stdio_init_all();

    // Initialize and turn on the onboard LED
    const uint LED_PIN = PICO_DEFAULT_LED_PIN;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    // gpio_put(LED_PIN, 1); 

    while (true) {
        // keep printing to stdio so you can also verify via serial
        printf("Miko Says: Hello, world!\n");
        gpio_put(LED_PIN, 1); 
        sleep_ms(1000);
        gpio_put(LED_PIN, 0); 
        printf("Miko Says: Thanks for listening!\n");
        gpio_put(LED_PIN, 1); 
        sleep_ms(500);
        gpio_put(LED_PIN, 0); 
        printf("Miko Says: Hope to work with you soon :)\n");
        sleep_ms(500);
        gpio_put(LED_PIN, 1);
        sleep_ms(500);
        gpio_put(LED_PIN, 0); 
    }
}
