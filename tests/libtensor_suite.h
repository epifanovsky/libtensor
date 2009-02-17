#ifndef __LIBTENSOR_LIBTENSOR_SUITE_H
#define __LIBTENSOR_LIBTENSOR_SUITE_H

#include <libtest.h>
#include <libtensor.h>
#include "permutation_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/**
	\brief Test suite for the tensor library (libtensor)

	This suite runs the following tests:
	\li libtensor::permutation_test<permutation>
	\li libtensor::permutation_test<permutation_lehmer>
**/
class libtensor_suite : public libtest::test_suite {
private:
	unit_test_factory< permutation_test<permutation> > m_utf_permutation;
	unit_test_factory< permutation_test<permutation_lehmer> >
		m_utf_permutation_lehmer;

public:
	//!	Creates the suite
	libtensor_suite();
};

} // namespace libtensor

#endif // __LIBTENSOR_LIBTENSOR_SUITE_H

