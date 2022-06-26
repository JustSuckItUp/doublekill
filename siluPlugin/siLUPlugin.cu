/*
 silu plugin
            x
            |
   ---------|
   |        |
   sigmoid  |
   |        |
   ---------|
            Mul
            |
 */
 
#include "siLUPlugin.h"

using namespace nvinfer1;

PluginFieldCollection SiLUPluginCreator::fc_{};
std::vector<PluginField> SiLUPluginCreator::attr_;

__global__ void siLUKernel(float *pInput, float *pOutput)
{
    const int tx = threadIdx.x, index = blockIdx.x * 256 + threadIdx.x;

    float ans_sig=1/(1+exp(-1*pInput[index])) ;
    __syncthreads();
    pOutput[index] = ans_sig*pInput[index];


}

int32_t SiLUPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];

    siLUKernel <<<nBlock, 128, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(SiLUPluginCreator);

