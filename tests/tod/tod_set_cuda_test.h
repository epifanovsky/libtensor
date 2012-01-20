#ifndef LIBTENSOR_TOD_CUDA_SET_TEST_H
#define LIBTENSOR_TOD_CUDA_SET_TEST_H

#include <libtest/unit_test.h>
#include <libvmm/cuda_allocator.cu>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_set_cuda class

	\ingroup libtensor_tests_tod
**/
class tod_set_cuda_test : public libtest::unit_test {
static const double k_thresh; //!< Threshold multiplier

public:
	virtual void perform() throw(libtest::test_exception);

	/**	\brief Copy tensor from Host to Device

		**/
		template<typename T, size_t N>
		void copyTensorHostToDevice(dense_tensor<N, T, std_allocator<T> > &ht, dense_tensor<N, T, libvmm::cuda_allocator<T> > &dt);


		/**	\brief Copy tensor from Device to Host

			**/
		template<typename T, size_t N>
		void copyTensorDeviceToHost(dense_tensor<N, T, libvmm::cuda_allocator<T> > &dt, dense_tensor<N, T, std_allocator<T> > &ht);

};

} // namespace libtensor

#endif // LIBTENSOR_TOD_CUDA_SET_TEST_H

