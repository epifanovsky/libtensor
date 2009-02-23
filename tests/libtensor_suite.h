#ifndef LIBTENSOR_LIBTENSOR_SUITE_H
#define LIBTENSOR_LIBTENSOR_SUITE_H

#include <libtest.h>
#include <libtensor.h>
#include "default_symmetry_test.h"
#include "dimensions_test.h"
#include "index_test.h"
#include "index_range_test.h"
#include "lehmer_code_test.h"
#include "permutation_test.h"
#include "permutator_test.h"
#include "tensor_test.h"
#include "tod_set_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/**
	\brief Test suite for the tensor library (libtensor)

	This suite runs the following tests:
	\li libtensor::default_symmetry_test
	\li libtensor::permutation_test
	\li libtensor::lehmer_code_test
	\li libtensor::permutator_test
	\li libtensor::index_test
	\li libtensor::index_range_test
	\li libtensor::dimensions_test
	\li libtensor::tensor_test
	\li libtensor::tod_set_test
**/
class libtensor_suite : public libtest::test_suite {
private:
	unit_test_factory< default_symmetry_test > m_utf_default_symmetry;
	unit_test_factory< permutation_test > m_utf_permutation;
	unit_test_factory< lehmer_code_test > m_utf_lehmer_code;
	unit_test_factory< permutator_test > m_utf_permutator;
	unit_test_factory< index_test > m_utf_index;
	unit_test_factory< index_range_test > m_utf_index_range;
	unit_test_factory< dimensions_test > m_utf_dimensions;
	unit_test_factory< tensor_test > m_utf_tensor;
	unit_test_factory< tod_set_test > m_utf_tod_set;

public:
	//!	Creates the suite
	libtensor_suite();
};

} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_SUITE_H

