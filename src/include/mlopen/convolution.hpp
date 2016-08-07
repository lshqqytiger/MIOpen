#ifndef _MLOPEN_CONVOLUTION_HPP_
#define _MLOPEN_CONVOLUTION_HPP_

#include <mlopen.h>
#include <mlopen/handle.hpp>
#include <mlopen/tensor.hpp>
#include <mlopen/kernel_cache.hpp>
#include <mlopen/common.hpp>

struct mlopenConvolutionDescriptor {
	
	mlopenConvolutionDescriptor();
	~mlopenConvolutionDescriptor() {}

	mlopenStatus_t GetForwardOutputDim(const mlopen::TensorDescriptor& inputTensorDesc,
			const mlopen::TensorDescriptor& filterDesc,
			int *n,
			int *c,
			int *h, 
			int *w);

	mlopenStatus_t FindConvFwdAlgorithm(mlopenHandle_t handle,
		const mlopen::TensorDescriptor&	xDesc,
		const Data_t					x,
		const mlopen::TensorDescriptor&	wDesc,
		const Data_t					w,
		const mlopen::TensorDescriptor&	yDesc,
		const Data_t					y,
		const int						requestAlgoCount,
		int								*returnedAlgoCount,
		mlopenConvAlgoPerf_t			*perfResults,
		mlopenConvPreference_t			preference,
		void							*workSpace,
		size_t							workSpaceSize,
		bool							exhaustiveSearch);

	mlopenStatus_t ConvolutionForward(mlopenHandle_t handle,
		const void							*alpha,
		const mlopen::TensorDescriptor&		xDesc,
		const Data_t						x,
		const mlopen::TensorDescriptor&		wDesc,
		const Data_t						w,
		mlopenConvFwdAlgorithm_t			algo,
		const void							*beta,
		const mlopen::TensorDescriptor&		yDesc,
		Data_t								y,
		void								*workSpace,
		size_t								workSpaceSize);

	mlopenStatus_t FindConvBwdDataAlgorithm(mlopenHandle_t handle,
		const mlopen::TensorDescriptor&	dyDesc,
		const Data_t					dy,
		const mlopen::TensorDescriptor&	wDesc,
		const Data_t					w,
		const mlopen::TensorDescriptor&	dxDesc,
		const Data_t					dx,
		const int						requestAlgoCount,
		int								*returnedAlgoCount,
		mlopenConvAlgoPerf_t			*perfResults,
		mlopenConvPreference_t			preference,
		void							*workSpace,
		size_t							workSpaceSize);

	mlopenStatus_t ConvolutionBackwardData(mlopenHandle_t handle,
		const void							*alpha,
		const mlopen::TensorDescriptor&		dyDesc,
		const Data_t						dy,
		const mlopen::TensorDescriptor&		wDesc,
		const Data_t						w,
		mlopenConvBwdDataAlgorithm_t		algo,
		const void							*beta,
		const mlopen::TensorDescriptor&		dxDesc,
		Data_t								dx,
		void								*workSpace,
		size_t								workSpaceSize);

	mlopenConvolutionMode_t _mode;
	int _pad_h;
	int _pad_w;
	int _u;
	int _v;
	int _upscalex;
	int _upscaley;
};

#endif // _MLOPEN_CONVOLUTION_HPP_
