#ifndef LIBTENSOR_LIBTENSOR_EXPR_SUITE_H
#define LIBTENSOR_LIBTENSOR_EXPR_SUITE_H

#include <libtest/test_suite.h>
#include "eval_plan_test.h"
#include "expr_tree_test.h"
#include "graph_test.h"
#include "node_add_test.h"
#include "node_contract_test.h"
#include "node_diag_test.h"
#include "node_dot_product_test.h"
#include "node_ident_test.h"
#include "node_product_test.h"
#include "node_transform_test.h"

using libtest::unit_test_factory;

namespace libtensor {


/** \defgroup libtensor_tests_expr Tests of the expression optimizer
    \ingroup libtensor_tests
 **/

/** \brief Test suite for the expression optimizer of libtensor

    This suite runs the following tests:
     - libtensor::eval_plan_test
     - libtensor::expr_tree_test
     - libtensor::graph_test
     - libtensor::node_add_test
     - libtensor::node_contract_test
     - libtensor::node_diag_test
     - libtensor::node_dot_product_test
     - libtensor::node_ewmult_test
     - libtensor::node_ident_test
     - libtensor::node_mult_test
     - libtensor::node_product_test
     - libtensor::node_transform_test

    \ingroup libtensor_tests_expr
 **/
class libtensor_expr_suite : public libtest::test_suite {
private:
//    unit_test_factory<eval_plan_test> m_utf_eval_plan;
    unit_test_factory<expr_tree_test> m_utf_expr_tree;
    unit_test_factory<graph_test> m_utf_graph;
    unit_test_factory<node_add_test> m_utf_node_add;
    unit_test_factory<node_contract_test> m_utf_node_contract;
    unit_test_factory<node_diag_test> m_utf_node_diag;
    unit_test_factory<node_dot_product_test> m_utf_node_dot_product;
    unit_test_factory<node_ident_test> m_utf_node_ident;
    unit_test_factory<node_product_test> m_utf_node_product;
    unit_test_factory<node_transform_test> m_utf_node_transform;

public:
    //! Creates the suite
    libtensor_expr_suite();

};


} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_EXPR_SUITE_H
