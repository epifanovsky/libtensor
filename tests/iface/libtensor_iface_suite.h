#ifndef LIBTENSOR_LIBTENSOR_IFACE_SUITE_H
#define LIBTENSOR_LIBTENSOR_IFACE_SUITE_H

#include <libtest/test_suite.h>
#include "anon_eval_test.h"
#include "any_tensor_test.h"
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
#include "eval_btensor_double_test.h"
#include "ewmult_test.h"
#include "expr_test.h"
#include "expr_tensor_test.h"
//#include "labeled_btensor_test.h"
#include "letter_expr_test.h"
#include "letter_test.h"
#include "mult_test.h"
#include "symm_test.h"
#include "tensor_list_test.h"
#include "trace_test.h"

using libtest::unit_test_factory;

namespace libtensor {


/** \defgroup libtensor_tests_iface Tests of the easy-to-use interface
    \brief Unit tests of the easy-to-use interface of libtensor
    \ingroup libtensor_tests
 **/

/** \brief Test suite for the easy-to-use interface of libtensor

    This suite runs the following tests:
     - libtensor::anon_eval_test
     - libtensor::any_tensor_test
     - libtensor::bispace_test
     - libtensor::bispace_expr_test
     - libtensor::btensor_test
     - libtensor::contract_test
     - libtensor::diag_test
     - libtensor::direct_btensor_test
     - libtensor::direct_eval_test
     - libtensor::direct_product_test
     - libtensor::dirsum_test
     - libtensor::dot_product_test
     - libtensor::eval_btensor_double_test
     - libtensor::ewmult_test
     - libtensor::expr_test
     - libtensor::expr_tensor_test
     ------ libtensor::labeled_btensor_test
     - libtensor::letter_test
     - libtensor::letter_expr_test
     - libtensor::mult_test
     - libtensor::symm_test
     - libtensor::tensor_list_test
     - libtensor::trace_test

    \ingroup libtensor_tests
 **/
class libtensor_iface_suite : public libtest::test_suite {
private:
    unit_test_factory<anon_eval_test> m_utf_anon_eval;
    unit_test_factory<any_tensor_test> m_utf_any_tensor;
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
    unit_test_factory<eval_btensor_double_test> m_utf_eval_btensor_double;
    unit_test_factory<ewmult_test> m_utf_ewmult;
    unit_test_factory<expr_test> m_utf_expr;
    unit_test_factory<expr_tensor_test> m_utf_expr_tensor;
//    unit_test_factory<labeled_btensor_test> m_utf_labeled_btensor;
    unit_test_factory<letter_test> m_utf_letter;
    unit_test_factory<letter_expr_test> m_utf_letter_expr;
    unit_test_factory<mult_test> m_utf_mult;
    unit_test_factory<symm_test> m_utf_symm;
    unit_test_factory<tensor_list_test> m_utf_tensor_list;
    unit_test_factory<trace_test> m_utf_trace;

public:
    //! Creates the suite
    libtensor_iface_suite();

};


} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_IFACE_SUITE_H

