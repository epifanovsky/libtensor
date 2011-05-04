#ifndef LIBTENSOR_LIBTENSOR_IFACE_SUITE_H
#define LIBTENSOR_LIBTENSOR_IFACE_SUITE_H

#include <libtest/test_suite.h>
#include "anon_eval_test.h"
#include "bispace_test.h"
#include "bispace_expr_test.h"
#include "btensor_test.h"
#include "contract_test.h"
#include "diag_test.h"
#include "direct_btensor_test.h"
#include "direct_eval_test.h"
#include "direct_product_test.h"
#include "dirsum_test.h"
#include "dot_product_test.h"
#include "expr_test.h"
#include "labeled_btensor_test.h"
#include "letter_expr_test.h"
#include "letter_test.h"
#include "mapped_btensor_test.h"
#include "mult_test.h"
#include "symm_test.h"
#include "trace_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/** \defgroup libtensor_tests_iface Tests of the easy-to-use interface
 	\brief Unit tests of the easy-to-use interface of libtensor
 	\ingroup libtensor_tests
 **/

/**
	\brief Test suite for the easy-to-use interface of libtensor
	\ingroup libtensor_tests

	This suite runs the following tests:
	\li libtensor::anon_eval_test
	\li libtensor::bispace_test
	\li libtensor::bispace_expr_test
	\li libtensor::btensor_test
	\li libtensor::contract_test
	\li libtensor::diag_test
	\li libtensor::direct_btensor_test
	\li libtensor::direct_eval_test
	\li libtensor::direct_product_test
	\li libtensor::dirsum_test
	\li libtensor::dot_product_test
	\li libtensor::expr_test
	\li libtensor::labeled_btensor_test
	\li libtensor::letter_test
	\li libtensor::letter_expr_test
	\li libtensor::mapped_btensor_test
	\li libtensor::mult_test
	\li libtensor::symm_test
	\li libtensor::trace_test
**/
class libtensor_iface_suite : public libtest::test_suite {
private:
	unit_test_factory<anon_eval_test> m_utf_anon_eval;
	unit_test_factory<bispace_test> m_utf_bispace;
	unit_test_factory<bispace_expr_test> m_utf_bispace_expr;
	unit_test_factory<btensor_test> m_utf_btensor;
	unit_test_factory<contract_test> m_utf_contract;
	unit_test_factory<diag_test> m_utf_diag;
	unit_test_factory<direct_btensor_test> m_utf_direct_btensor;
	unit_test_factory<direct_eval_test> m_utf_direct_eval;
	unit_test_factory<direct_product_test> m_utf_direct_product;
	unit_test_factory<dirsum_test> m_utf_dirsum;
	unit_test_factory<dot_product_test> m_utf_dot_product;
	unit_test_factory<expr_test> m_utf_expr;
	unit_test_factory<labeled_btensor_test> m_utf_labeled_btensor;
	unit_test_factory<letter_test> m_utf_letter;
	unit_test_factory<letter_expr_test> m_utf_letter_expr;
	unit_test_factory<mapped_btensor_test> m_utf_mapped_btensor;
	unit_test_factory<mult_test> m_utf_mult;
	unit_test_factory<symm_test> m_utf_symm;
	unit_test_factory<trace_test> m_utf_trace;

public:
	//!	Creates the suite
	libtensor_iface_suite();
};

} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_IFACE_SUITE_H

