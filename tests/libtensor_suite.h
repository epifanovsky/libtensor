#ifndef LIBTENSOR_LIBTENSOR_SUITE_H
#define LIBTENSOR_LIBTENSOR_SUITE_H

#include <libtest.h>
#include <libtensor.h>
#include "bispace_test.h"
#include "btensor_test.h"
#include "contract2_0_4i_test.h"
#include "default_symmetry_test.h"
#include "dimensions_test.h"
#include "direct_tensor_test.h"
#include "expr_binary_test.h"
#include "expr_identity_test.h"
#include "expr_literal_test.h"
#include "expression_test.h"
#include "immutable_test.h"
#include "index_test.h"
#include "index_range_test.h"
#include "lehmer_code_test.h"
#include "letter_test.h"
#include "permutation_test.h"
#include "permutator_test.h"
#include "symmetry_test.h"
#include "tensor_test.h"
#include "tod_add_test.h"
#include "tod_compare_test.h"
#include "tod_contract2_impl_022_test.h"
#include "tod_contract2_impl_113_test.h"
#include "tod_contract2_impl_131_test.h"
#include "tod_contract2_impl_221_test.h"
#include "tod_contract2_test.h"
#include "tod_copy_test.h"
#include "tod_set_test.h"
#include "tod_sum_test.h"
#include "tod_symcontract2_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/**
	\brief Test suite for the tensor library (libtensor)

	This suite runs the following tests:
	\li libtensor::bispace_test
	\li libtensor::btensor_test
	\li libtensor::contract2_0_4i_test
	\li libtensor::default_symmetry_test
	\li libtensor::dimensions_test
	\li libtensor::direct_tensor_test
	\li libtensor::expr_binary_test
	\li libtensor::expr_identity_test
	\li libtensor::expr_literal_test
	\li libtensor::expression_test
	\li libtensor::immutable_test
	\li libtensor::index_test
	\li libtensor::index_range_test
	\li libtensor::lehmer_code_test
	\li libtensor::letter_test
	\li libtensor::permutation_test
	\li libtensor::permutator_test
	\li libtensor::symmetry_test
	\li libtensor::tensor_test
	\li libtensor::tod_add_test
	\li libtensor::tod_compare_test
	\li libtensor::tod_contract2_test
	\li libtensor::tod_contract2_impl_022_test
	\li libtensor::tod_contract2_impl_113_test
	\li libtensor::tod_contract2_impl_131_test
	\li libtensor::tod_contract2_impl_221_test
	\li libtensor::tod_copy_test
	\li libtensor::tod_set_test
	\li libtensor::tod_sum_test
	\li libtensor::tod_symcontract2_test
**/
class libtensor_suite : public libtest::test_suite {
private:
	unit_test_factory<bispace_test> m_utf_bispace;
	unit_test_factory<btensor_test> m_utf_btensor;
	unit_test_factory<contract2_0_4i_test> m_utf_contract2_0_4i;
	unit_test_factory<default_symmetry_test> m_utf_default_symmetry;
	unit_test_factory<dimensions_test> m_utf_dimensions;
	unit_test_factory<direct_tensor_test> m_utf_direct_tensor;
	unit_test_factory<expr_binary_test> m_utf_expr_binary;
	unit_test_factory<expr_identity_test> m_utf_expr_identity;
	unit_test_factory<expr_literal_test> m_utf_expr_literal;
	unit_test_factory<expression_test> m_utf_expression;
	unit_test_factory<immutable_test> m_utf_immutable;
	unit_test_factory<index_test> m_utf_index;
	unit_test_factory<index_range_test> m_utf_index_range;
	unit_test_factory<lehmer_code_test> m_utf_lehmer_code;
	unit_test_factory<letter_test> m_utf_letter;
	unit_test_factory<permutation_test> m_utf_permutation;
	unit_test_factory<permutator_test> m_utf_permutator;
	unit_test_factory<symmetry_test> m_utf_symmetry;
	unit_test_factory<tensor_test> m_utf_tensor;
	unit_test_factory<tod_add_test> m_utf_tod_add;
	unit_test_factory<tod_compare_test> m_utf_tod_compare;
	unit_test_factory<tod_contract2_impl_022_test>
		m_utf_tod_contract2_impl_022;
	unit_test_factory<tod_contract2_impl_113_test>
		m_utf_tod_contract2_impl_113;
	unit_test_factory<tod_contract2_impl_131_test>
		m_utf_tod_contract2_impl_131;
	unit_test_factory<tod_contract2_impl_221_test>
		m_utf_tod_contract2_impl_221;
	unit_test_factory<tod_contract2_test> m_utf_tod_contract2;
	unit_test_factory<tod_copy_test> m_utf_tod_copy;
	unit_test_factory<tod_set_test> m_utf_tod_set;
	unit_test_factory<tod_sum_test> m_utf_tod_sum;
	unit_test_factory<tod_symcontract2_test> m_utf_tod_symcontract2;

public:
	//!	Creates the suite
	libtensor_suite();
};

} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_SUITE_H

