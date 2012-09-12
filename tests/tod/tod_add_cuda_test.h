#ifndef LIBTENSOR_TOD_ADD_CUDA_TEST_H
#define LIBTENSOR_TOD_ADD_CUDA_TEST_H

#include <libtest/unit_test.h>
//#include <libvmm/std_allocator.h>
#include <libvmm/cuda_allocator.h>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>

namespace libtensor {

/**	\brief Tests the libtensor::tod_add_cuda class

	\ingroup libtensor_tests_tod
**/
class tod_add_cuda_test : public libtest::unit_test {
	static const double k_thresh; //!< Threshold multiplier
public:
	virtual void perform() throw(libtest::test_exception);
private:
	/**	\brief Tests if exceptions are thrown when the tensors have
			different dimensions
	**/
	void test_exc() throw(libtest::test_exception);

	/**	\brief Tests addition of a tensor to itself

		\f[ T_{pqrs} = 2.0 A_{pqrs} + 0.5 A_{pqrs}  \f]
	**/
	void test_add_to_self_pqrs(size_t, size_t, size_t, size_t)
		throw(libtest::test_exception);

	/**	\brief Tests addition of two tensors (no permutation)

		\f[ T_{pqrs} = T_{pqrs} + 0.1 A_{pqrs}  \f]
	**/
	void test_add_two_pqrs_pqrs(size_t, size_t, size_t, size_t)
		throw(libtest::test_exception);

	/**	\brief Tests addition of two tensors (permutation type 1)

		\f[ T_{pqrs} = T_{pqrs} + 0.1 A_{qprs}  \f]
	**/
	void test_add_two_pqrs_qprs(size_t, size_t, size_t, size_t)
		throw(libtest::test_exception);

	/**	\brief Tests addition of two tensors (permutation type 2)

		\f[ T_{pqrs} = T_{pqrs} + 0.1 A_{prsq}  \f]
	**/
	void test_add_two_pqrs_prsq(size_t, size_t, size_t, size_t)
		throw(libtest::test_exception);

	/**	\brief Tests addition of two tensors (permutation type 3)

		\f[ T_{pqrs} = T_{pqrs} + 0.1 A_{qpsr}  \f]
	**/
	void test_add_two_pqrs_qpsr(size_t, size_t, size_t, size_t)
		throw(libtest::test_exception);

	/**	\brief Tests addition of three tensors

		\f[ T_{pqrs} = T_{pqrs} + 0.5 \left( A_{pqrs} - 4.0 B_{qprs} \right) \f]
	**/
	void test_add_mult(size_t, size_t, size_t, size_t)
		throw(libtest::test_exception);

	/**	\brief Tests addition of three tensors (in two dimensions)

		\f[ T_{pq} = T_{pq} + 0.5 \left( 2.0 A_{pq} - B_{qp} \right) \f]
	**/
	void test_add_two_pq_qp( size_t, size_t )
		throw(libtest::test_exception);

	void test_add_two_ijkl_kjli(size_t ni, size_t nj, size_t nk, size_t nl,
		double c1, double c2) throw(libtest::test_exception);

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

#endif // LIBTENSOR_TOD_ADD_CUDA_TEST_H

