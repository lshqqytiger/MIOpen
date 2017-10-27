/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <miopen/activ.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/float_equal.hpp>

namespace miopen {

miopenStatus_t ActivationDescriptor::Forward(Handle& handle,
                                             const void* alpha,
                                             const TensorDescriptor& xDesc,
                                             ConstData_t x,
                                             const void* beta,
                                             const TensorDescriptor& yDesc,
                                             Data_t y,
                                             size_t xOffset,
                                             size_t yOffset)
{

    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }
    miopenStatus_t status = miopenStatusSuccess;

    mlo_construct_neuron construct_params(1); // forward

    construct_params.setStream(&handle);

    int nOut;
    int cOut;
    int hOut;
    int wOut;
    int nOutStride;
    int cOutStride;
    int hOutStride;
    int wOutStride;

    std::tie(nOut, cOut, hOut, wOut)                         = tien<4>(yDesc.GetLengths());
    std::tie(nOutStride, cOutStride, hOutStride, wOutStride) = tien<4>(yDesc.GetStrides());

    construct_params.setTopDescr(
        "NCHW", "FP32", nOut, cOut, hOut, wOut, nOutStride, cOutStride, hOutStride, wOutStride);
    int nIn;
    int cIn;
    int hIn;
    int wIn;
    int nInStride;
    int cInStride;
    int hInStride;
    int wInStride;

    std::tie(nIn, cIn, hIn, wIn)                         = tien<4>(xDesc.GetLengths());
    std::tie(nInStride, cInStride, hInStride, wInStride) = tien<4>(xDesc.GetStrides());

    construct_params.setBotDescr(
        "NCHW", "FP32", nIn, cIn, hIn, wIn, nInStride, cInStride, hInStride, wInStride);

    double activ_alpha = GetAlpha();
    double activ_beta  = GetBeta();
    double activ_power = GetPower();

    construct_params.setNeuronDescr(static_cast<int>(mode), activ_power, activ_beta, activ_alpha);

    status = static_cast<miopenStatus_t>(construct_params.mloConstruct());

    std::string program_name     = construct_params.getKernelFile();      // CL kernel filename
    std::string kernel_name      = construct_params.getKernelName();      // kernel name
    std::string compiler_options = construct_params.getCompilerOptions(); // kernel parameters

    std::string network_config;
    construct_params.mloBuildConf_Key(network_config);

    const std::vector<size_t>& vld = construct_params.getLocalWkSize();
    const std::vector<size_t>& vgd = construct_params.getGlobalWkSize();

    int imode = mode;
    construct_params.getNeuronDescr(imode, activ_power, activ_beta, activ_alpha);
    auto f_activ_alpha = static_cast<float>(activ_alpha);
    auto f_activ_beta  = static_cast<float>(activ_beta);
    auto f_activ_power = static_cast<float>(activ_power);

    compiler_options +=
        " -DMLO_N_IN=" + std::to_string(nIn) + " -DMLO_C_IN=" + std::to_string(cIn) +
        " -DMLO_H_IN=" + std::to_string(hIn) + " -DMLO_W_IN=" + std::to_string(wIn) +
        " -DMLO_N_IN_STRIDE=" + std::to_string(nInStride) + " -DMLO_C_IN_STRIDE=" +
        std::to_string(cInStride) + " -DMLO_H_IN_STRIDE=" + std::to_string(hInStride) +
        " -DMLO_W_IN_STRIDE=" + std::to_string(wInStride) + " -DMLO_N_OUT=" + std::to_string(nOut) +
        " -DMLO_C_OUT=" + std::to_string(cOut) + " -DMLO_H_OUT=" + std::to_string(hOut) +
        " -DMLO_W_OUT=" + std::to_string(wOut) + " -DMLO_N_OUT_STRIDE=" +
        std::to_string(nOutStride) + " -DMLO_C_OUT_STRIDE=" + std::to_string(cOutStride) +
        " -DMLO_H_OUT_STRIDE=" + std::to_string(hOutStride) + " -DMLO_W_OUT_STRIDE=" +
        std::to_string(wOutStride) + " -DMLO_N_DIN=" + std::to_string(1) + " -DMLO_C_DIN=" +
        std::to_string(1) + " -DMLO_H_DIN=" + std::to_string(1) + " -DMLO_W_DIN=" +
        std::to_string(1) + " -DMLO_N_DIN_STRIDE=" + std::to_string(1) + " -DMLO_C_DIN_STRIDE=" +
        std::to_string(1) + " -DMLO_H_DIN_STRIDE=" + std::to_string(1) + " -DMLO_W_DIN_STRIDE=" +
        std::to_string(1) + " -DMLO_N_DOUT=" + std::to_string(1) + " -DMLO_C_DOUT=" +
        std::to_string(1) + " -DMLO_H_DOUT=" + std::to_string(1) + " -DMLO_W_DOUT=" +
        std::to_string(1) + " -DMLO_N_DOUT_STRIDE=" + std::to_string(1) + " -DMLO_C_DOUT_STRIDE=" +
        std::to_string(1) + " -DMLO_H_DOUT_STRIDE=" + std::to_string(1) + " -DMLO_W_DOUT_STRIDE=" +
        std::to_string(1) + " -DMLO_IN_BLOCK_SZ=" + std::to_string(cIn*hIn*wIn) + 
		" -DMLO_OUT_BLOCK_SZ=" + std::to_string(cOut*hOut*wOut) + " -DMLO_DIN_BLOCK_SZ=" +
		std::to_string(1) + " -DMLO_DOUT_BLOCK_SZ=" + std::to_string(1);

    handle.GetKernel("miopenActivationForward",
                     network_config,
                     program_name,
                     kernel_name,
                     vld,
                     vgd,
                     compiler_options)(
        x, y, f_activ_power, f_activ_beta, f_activ_alpha, long(xOffset), long(yOffset));

    return (status);
}

miopenStatus_t ActivationDescriptor::Backward(Handle& handle,
                                              const void* alpha,
                                              const TensorDescriptor& yDesc,
                                              ConstData_t y,
                                              const TensorDescriptor& dyDesc,
                                              ConstData_t dy,
                                              const TensorDescriptor& xDesc,
                                              ConstData_t x,
                                              const void* beta,
                                              const TensorDescriptor& dxDesc,
                                              Data_t dx,
                                              size_t yOffset,
                                              size_t dyOffset,
                                              size_t xOffset,
                                              size_t dxOffset)
{

    if(!float_equal(*(static_cast<const float*>(alpha)), 1.0) ||
       !float_equal(*(static_cast<const float*>(beta)), 0))
    {
        MIOPEN_THROW("Only alpha=1 and beta=0 is supported");
    }
    miopenStatus_t status = miopenStatusSuccess;

    mlo_construct_neuron construct_params(0); // backward

    construct_params.setStream(&handle);
    int ndOut;
    int cdOut;
    int hdOut;
    int wdOut;
    int ndOutStride;
    int cdOutStride;
    int hdOutStride;
    int wdOutStride;

    std::tie(ndOut, cdOut, hdOut, wdOut)                         = tien<4>(dyDesc.GetLengths());
    std::tie(ndOutStride, cdOutStride, hdOutStride, wdOutStride) = tien<4>(dyDesc.GetStrides());

    construct_params.setTopDfDescr("NCHW",
                                   "FP32",
                                   ndOut,
                                   cdOut,
                                   hdOut,
                                   wdOut,
                                   ndOutStride,
                                   cdOutStride,
                                   hdOutStride,
                                   wdOutStride);

    int nOut;
    int cOut;
    int hOut;
    int wOut;
    int nOutStride;
    int cOutStride;
    int hOutStride;
    int wOutStride;

    std::tie(nOut, cOut, hOut, wOut)                         = tien<4>(yDesc.GetLengths());
    std::tie(nOutStride, cOutStride, hOutStride, wOutStride) = tien<4>(yDesc.GetStrides());

    construct_params.setTopDescr(
        "NCHW", "FP32", nOut, cOut, hOut, wOut, nOutStride, cOutStride, hOutStride, wOutStride);

    int ndIn;
    int cdIn;
    int hdIn;
    int wdIn;
    int ndInStride;
    int cdInStride;
    int hdInStride;
    int wdInStride;

    std::tie(ndIn, cdIn, hdIn, wdIn)                         = tien<4>(dxDesc.GetLengths());
    std::tie(ndInStride, cdInStride, hdInStride, wdInStride) = tien<4>(dxDesc.GetStrides());

    construct_params.setBotDfDescr(
        "NCHW", "FP32", ndIn, cdIn, hdIn, wdIn, ndInStride, cdInStride, hdInStride, wdInStride);

    int nIn;
    int cIn;
    int hIn;
    int wIn;
    int nInStride;
    int cInStride;
    int hInStride;
    int wInStride;

    std::tie(nIn, cIn, hIn, wIn)                         = tien<4>(xDesc.GetLengths());
    std::tie(nInStride, cInStride, hInStride, wInStride) = tien<4>(xDesc.GetStrides());

    construct_params.setBotDescr(
        "NCHW", "FP32", nIn, cIn, hIn, wIn, nInStride, cInStride, hInStride, wInStride);

    int activ_mode     = GetMode();
    double activ_alpha = GetAlpha();
    double activ_beta  = GetBeta();
    double activ_power = GetPower();

    construct_params.setNeuronDescr(activ_mode, activ_power, activ_beta, activ_alpha);

    status = static_cast<miopenStatus_t>(construct_params.mloConstruct());

    std::string program_name     = construct_params.getKernelFile();      // CL kernel filename
    std::string kernel_name      = construct_params.getKernelName();      // kernel name
    std::string compiler_options = construct_params.getCompilerOptions(); // kernel parameters

    std::string network_config;
    construct_params.mloBuildConf_Key(network_config);

    const std::vector<size_t>& vld = construct_params.getLocalWkSize();
    const std::vector<size_t>& vgd = construct_params.getGlobalWkSize();

    auto f_activ_alpha = static_cast<float>(GetAlpha());
    auto f_activ_beta  = static_cast<float>(GetBeta());
    auto f_activ_power = static_cast<float>(GetPower());
    float f_diff_scale = f_activ_beta * f_activ_power;

    compiler_options +=
        " -DMLO_N_IN=" + std::to_string(nIn) + " -DMLO_C_IN=" + std::to_string(cIn) +
        " -DMLO_H_IN=" + std::to_string(hIn) + " -DMLO_W_IN=" + std::to_string(wIn) +
        " -DMLO_N_IN_STRIDE=" + std::to_string(nInStride) + " -DMLO_C_IN_STRIDE=" +
        std::to_string(cInStride) + " -DMLO_H_IN_STRIDE=" + std::to_string(hInStride) +
        " -DMLO_W_IN_STRIDE=" + std::to_string(wInStride) + " -DMLO_N_OUT=" + std::to_string(nOut) +
        " -DMLO_C_OUT=" + std::to_string(cOut) + " -DMLO_H_OUT=" + std::to_string(hOut) +
        " -DMLO_W_OUT=" + std::to_string(wOut) + " -DMLO_N_OUT_STRIDE=" +
        std::to_string(nOutStride) + " -DMLO_C_OUT_STRIDE=" + std::to_string(cOutStride) +
        " -DMLO_H_OUT_STRIDE=" + std::to_string(hOutStride) + " -DMLO_W_OUT_STRIDE=" +
        std::to_string(wOutStride) + " -DMLO_N_DIN=" + std::to_string(ndIn) + " -DMLO_C_DIN=" +
        std::to_string(cdIn) + " -DMLO_H_DIN=" + std::to_string(hdIn) + " -DMLO_W_DIN=" +
        std::to_string(wdIn) + " -DMLO_N_DIN_STRIDE=" + std::to_string(ndInStride) +
        " -DMLO_C_DIN_STRIDE=" + std::to_string(cdInStride) + " -DMLO_H_DIN_STRIDE=" +
        std::to_string(hdInStride) + " -DMLO_W_DIN_STRIDE=" + std::to_string(wdInStride) +
        " -DMLO_N_DOUT=" + std::to_string(ndOut) + " -DMLO_C_DOUT=" + std::to_string(cdOut) +
        " -DMLO_H_DOUT=" + std::to_string(hdOut) + " -DMLO_W_DOUT=" + std::to_string(wdOut) +
        " -DMLO_N_DOUT_STRIDE=" + std::to_string(ndOutStride) + " -DMLO_C_DOUT_STRIDE=" +
        std::to_string(cdOutStride) + " -DMLO_H_DOUT_STRIDE=" + std::to_string(hdOutStride) +
        " -DMLO_W_DOUT_STRIDE=" + std::to_string(wdOutStride) + " -DMLO_IN_BLOCK_SZ=" + 
		std::to_string(cIn*hIn*wIn) + " -DMLO_OUT_BLOCK_SZ=" + std::to_string(cOut*hOut*wOut) + 
		" -DMLO_DIN_BLOCK_SZ=" +	std::to_string(cdIn*hdIn*wdIn) + " -DMLO_DOUT_BLOCK_SZ=" + 
		std::to_string(cdOut*hdOut*wdOut);

    handle.GetKernel("miopenActivationBackward",
                     network_config,
                     program_name,
                     kernel_name,
                     vld,
                     vgd,
                     compiler_options)(dx,
                                       dy,
                                       x,
                                       y,
                                       f_diff_scale,
                                       f_activ_power,
                                       f_activ_beta,
                                       f_activ_alpha,
                                       long(dxOffset),
                                       long(dyOffset),
                                       long(xOffset),
                                       long(yOffset));

    return (status);
}
} // namespace miopen
