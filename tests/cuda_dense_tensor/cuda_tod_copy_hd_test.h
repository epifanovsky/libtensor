#ifndef LIBTENSOR_CUDA_TOD_COPY_HD_TEST_H
#define LIBTENSOR_CUDA_TOD_COPY_HD_TEST_H

#include <libtest/unit_test.h>
#include <libtensor/core/dimensions.h>

namespace libtensor {

/**	\brief Tests the libtensor::cuda_tod_copy_h2d and libtensor::cuda_tod_copy_d2h classes

	\ingroup libtensor_tests_tod
**/
class cuda_tod_copy_hd_test : public libtest::unit_test {
public:
	virtual void perform() throw(libtest::test_exception);

private:
	/**	\brief Tests plain copying of a %tensor from host to device and back
	 **/
	template<size_t N>
	void test_plain(const dimensions<N> &dims)
		throw(libtest::test_exception);

	/**	\brief Tests if an exception is throws when the tensors have
			different dimensions
	 **/
	void test_exc() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_CUDA_TOD_COPY_HD_TEST_H

