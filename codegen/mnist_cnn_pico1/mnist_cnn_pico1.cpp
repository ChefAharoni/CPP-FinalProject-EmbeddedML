// mnist_cnn_pico1.cpp
// Auto-generated inference model implementation
// Pure C++ implementation for embedded systems

#include "mnist_cnn_pico1.h"
#include "mnist_cnn_pico1_weights.cpp"
#include "../../components/fully_connected.h"


#include "../../components/conv_2d.h"


#include "../../components/max_pool_2d.h"


#include "../../components/shape.h"


#include "../../components/strided_slice.h"


#include "../../components/pack.h"


#include "../../components/reshape.h"




#include "../../components/softmax.h"
#include <cstddef>

namespace embedded_ml {

void mnist_cnn_pico1Model::Inference(const float* input, float* output) {

    // Layer 0: CONV_2D




    Conv2D(input, weight_10_MNIST_Model_1_conv0_1_convolution, weight_11_MNIST_Model_1_activation_1_1_Relu_MNIST_Model_1_bn0_1_batchnorm_add_1_MNIST_Model_1_bn0_1_batchnorm_mul_1__MNIST_Model_1_conv0_1_BiasAdd_MNIST_Model_1_bn0_1_batchnorm_mul_MNIST_Model_1_conv0_1_convolution_MNIST_Model_1_bn0_1_batchnorm_sub, buffer_16, 1, 28, 28, 1, 3, 3, 32, 1, 1, PaddingType::VALID, ActivationType::RELU, 1, 1);





    // Layer 1: CONV_2D




    Conv2D(buffer_16, weight_7_arith_constant7, weight_3_arith_constant3, buffer_17, 1, 26, 26, 32, 3, 3, 32, 1, 1, PaddingType::VALID, ActivationType::RELU, 1, 1);





    // Layer 2: MAX_POOL_2D





    MaxPool2D(buffer_17, buffer_18, 1, 24, 24, 32, 2, 2, 2, 2, PaddingType::VALID, ActivationType::NONE);






    // Layer 3: CONV_2D




    Conv2D(buffer_18, weight_6_arith_constant6, weight_2_arith_constant2, buffer_19, 1, 12, 12, 32, 3, 3, 64, 1, 1, PaddingType::VALID, ActivationType::RELU, 1, 1);





    // Layer 4: CONV_2D




    Conv2D(buffer_19, weight_5_arith_constant5, weight_1_arith_constant1, buffer_20, 1, 10, 10, 64, 3, 3, 64, 1, 1, PaddingType::VALID, ActivationType::RELU, 1, 1);





    // Layer 5: MAX_POOL_2D





    MaxPool2D(buffer_20, buffer_21, 1, 8, 8, 64, 2, 2, 2, 2, PaddingType::VALID, ActivationType::NONE);






    // Layer 6: SHAPE






    // SHAPE: Extract shape from input tensor
    {
        int32_t input_shape[4] = { 1, 4, 4, 64 };
        Shape(input_shape, 4, reinterpret_cast<int32_t*>(buffer_22));
    }







    // Layer 7: STRIDED_SLICE







    // STRIDED_SLICE: Extract slice from input
    // Note: This is a simplified implementation
    // Full implementation would extract begin/end/strides from input tensors
    // For now, copying input to output as placeholder
    for (size_t j = 0; j < 1; ++j) {
        buffer_23[j] = buffer_22[j];
    }








    // Layer 8: PACK








    // PACK: Pack multiple inputs along axis 0
    // Note: This is a simplified implementation
    // Full implementation would handle multiple input tensors
    // Copying first input to output as placeholder
    for (size_t j = 0; j < 2; ++j) {
        buffer_24[j] = buffer_23[j];
    }









    // Layer 9: RESHAPE









    Reshape(buffer_21, 1024, buffer_25, 1024);










    // Layer 10: FULLY_CONNECTED

    FullyConnected(buffer_25, weight_9_arith_constant9, weight_0_arith_constant, buffer_26, 1024, 256, ActivationType::RELU);


    // Layer 11: FULLY_CONNECTED

    FullyConnected(buffer_26, weight_8_arith_constant8, weight_4_arith_constant4, buffer_27, 256, 10, ActivationType::NONE);


    // Layer 12: SOFTMAX



    Softmax(buffer_27, output, 10);




}


// Intermediate buffer definitions

float mnist_cnn_pico1Model::buffer_16[21632];


float mnist_cnn_pico1Model::buffer_17[18432];


float mnist_cnn_pico1Model::buffer_18[4608];


float mnist_cnn_pico1Model::buffer_19[6400];


float mnist_cnn_pico1Model::buffer_20[4096];


float mnist_cnn_pico1Model::buffer_21[1024];


float mnist_cnn_pico1Model::buffer_22[4];


float mnist_cnn_pico1Model::buffer_23[1];


float mnist_cnn_pico1Model::buffer_24[2];


float mnist_cnn_pico1Model::buffer_25[1024];


float mnist_cnn_pico1Model::buffer_26[256];


float mnist_cnn_pico1Model::buffer_27[10];


float mnist_cnn_pico1Model::buffer_11[1];


float mnist_cnn_pico1Model::buffer_12[1];


float mnist_cnn_pico1Model::buffer_13[1];



} // namespace embedded_ml

