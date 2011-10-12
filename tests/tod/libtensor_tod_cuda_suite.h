#ifndef LIBTENSOR_LIBTENSOR_TOD_CUDA_SUITE_H
#define LIBTENSOR_LIBTENSOR_TOD_CUDA_SUITE_H

#include <libtest/test_suite.h>
#include "tod_cuda_copy_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/** \defgroup libtensor_tests_tod Tests of CUDA tensor operations
 	\brief Unit tests of the %tensor operations in libtensor
 	\ingroup libtensor_tests
 **/


/**
	\brief Test suite for the tensor operations in libtensor
	\ingroup libtensor_tests

	This suite runs the following tests:
	\li libtensor::tod_cuda_copy_test

**/
class libtensor_tod_cuda_suite : public libtest::test_suite {
private:
	unit_test_factory<tod_cuda_copy_test> m_utf_tod_cuda_copy;

public:
	//!	Creates the suite
	libtensor_tod_cuda_suite();
};

} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_SUITE_H

