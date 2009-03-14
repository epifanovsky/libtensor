#include <libtensor.h>
#include "libtensor_suite.h"

namespace libtensor {

libtensor_suite::libtensor_suite() : libtest::test_suite("libtensor") {
	add_test("contract2_2_2i", m_utf_contract2_2_2i);
	add_test("contract2_2_3i", m_utf_contract2_2_3i);
	add_test("default_symmetry", m_utf_default_symmetry);
	add_test("permutation", m_utf_permutation);
	//add_test("lehmer_code", m_utf_lehmer_code);
	add_test("permutator", m_utf_permutator);
	add_test("immutable", m_utf_immutable);
	add_test("index", m_utf_index);
	add_test("index_range", m_utf_index_range);
	add_test("dimensions", m_utf_dimensions);
	add_test("symmetry", m_utf_symmetry);
	add_test("tensor", m_utf_tensor);
	add_test("direct_tensor", m_utf_direct_tensor);
	add_test("tod_add", m_utf_tod_add);
	add_test("tod_contract2", m_utf_tod_contract2);
	add_test("tod_set", m_utf_tod_set);
	add_test("tod_sum", m_utf_tod_sum);
}

}

