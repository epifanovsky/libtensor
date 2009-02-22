#include <libtensor.h>
#include "libtensor_suite.h"

namespace libtensor {

libtensor_suite::libtensor_suite() : libtest::test_suite("libtensor") {
	add_test("default_symmetry", m_utf_default_symmetry);
	add_test("permutation", m_utf_permutation);
//	add_test("permutation_lehmer", m_utf_permutation_lehmer);
	add_test("lehmer_code", m_utf_lehmer_code);
	add_test("permutator", m_utf_permutator);
	add_test("index", m_utf_index);
	add_test("index_range", m_utf_index_range);
	add_test("dimensions", m_utf_dimensions);
	add_test("tensor", m_utf_tensor);
	add_test("tod_set", m_utf_tod_set);
}

}

