#include <libtensor.h>
#include "libtensor_suite.h"

namespace libtensor {

libtensor_suite::libtensor_suite() : libtest::test_suite("libtensor") {
	add_test("block_tensor", m_utf_block_tensor);
	add_test("contract2_0_4i", m_utf_contract2_0_4i);
	add_test("default_symmetry", m_utf_default_symmetry);
	add_test("dimensions", m_utf_dimensions);
	add_test("expr_binary", m_utf_expr_binary);
	add_test("expr_identity", m_utf_expr_identity);
	add_test("expr_literal", m_utf_expr_literal);
	add_test("expression", m_utf_expression);
	add_test("immutable", m_utf_immutable);
	add_test("index", m_utf_index);
	add_test("index_range", m_utf_index_range);
	add_test("permutation", m_utf_permutation);
	//add_test("lehmer_code", m_utf_lehmer_code);
	add_test("letter", m_utf_letter);
	add_test("permutator", m_utf_permutator);
	add_test("symmetry", m_utf_symmetry);
	add_test("tensor", m_utf_tensor);
	add_test("direct_tensor", m_utf_direct_tensor);
	add_test("tod_add", m_utf_tod_add);
	add_test("tod_contract2", m_utf_tod_contract2);
	add_test("tod_contract2_impl_022", m_utf_tod_contract2_impl_022);
	add_test("tod_contract2_impl_113", m_utf_tod_contract2_impl_113);
	add_test("tod_contract2_impl_131", m_utf_tod_contract2_impl_131);
	add_test("tod_set", m_utf_tod_set);
	add_test("tod_sum", m_utf_tod_sum);
	add_test("tod_symcontract2", m_utf_tod_symcontract2);
}

}

