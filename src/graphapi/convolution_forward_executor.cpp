/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <miopen/errors.hpp>
#include <miopen/fusion.hpp>
#include <miopen/graphapi/convolution_forward_executor.hpp>
#include <miopen/handle.hpp>
#include <miopen/visit_float.hpp>
#include <nlohmann/json.hpp>

namespace miopen {

namespace graphapi {

namespace {
std::vector<int> Convert(const std::vector<int64_t>& values)
{
    std::vector<int> converted(values.size());
    std::transform(values.begin(), values.end(), converted.begin(), [](int64_t value) {
        assert(value <= std::numeric_limits<int>::max() &&
               value >= std::numeric_limits<int>::min());
        return static_cast<int>(value);
    });

    return converted;
}

ConvolutionDescriptor Convert(const Convolution& conv, int groupCount)
{
    return {conv.getSpatialDims(),
            conv.getMode(),
            miopenPaddingMode_t::miopenPaddingDefault,
            Convert(conv.getPrePaddings()),
            Convert(conv.getFilterStrides()),
            Convert(conv.getDilations()),
            Convert(conv.getPostPaddings()),
            groupCount};
}
} // namespace

ConvolutionForwardExecutor::ConvolutionForwardExecutor(const nlohmann::json& json)
    : GraphPatternExecutor(),
      mXTensor(json.at(JsonFields::XTensor)),
      mWTensor(json.at(JsonFields::WTensor)),
      mConvolution(json.at(JsonFields::Convolution)),
      mGroupCount(json.at(JsonFields::GroupCount)),
      mYTensor(json.at(JsonFields::YTensor)),
      mAlpha(json.at(JsonFields::Alpha)),
      mBeta(json.at(JsonFields::Beta))
{
}

void ConvolutionForwardExecutor::execute(miopenHandle_t handle, const VariantPack& vpk)
{
    auto convDesc = Convert(mConvolution, mGroupCount);

    auto* xData = vpk.getDataPointer(mXTensor.getId());
    auto* wData = vpk.getDataPointer(mWTensor.getId());
    auto* yData = vpk.getDataPointer(mYTensor.getId());

    convDesc.ConvolutionForward(miopen::deref(handle),
                                &mAlpha,
                                mXTensor,
                                xData,
                                mWTensor,
                                wData,
                                miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoImplicitGEMM,
                                &mBeta,
                                mYTensor,
                                yData,
                                nullptr,
                                0);
}

nlohmann::json ConvolutionForwardExecutor::getJson()
{
    return {
        {GraphPatternExecutor::JsonFields::Name, name},
        {JsonFields::XTensor, mXTensor},
        {JsonFields::WTensor, mWTensor},
        {JsonFields::Convolution, mConvolution},
        {JsonFields::GroupCount, mGroupCount},
        {JsonFields::YTensor, mYTensor},
        {JsonFields::Alpha, mAlpha},
        {JsonFields::Beta, mBeta},
    };
}

} // namespace graphapi

} // namespace miopen
