#ifndef LIBTENSOR_LIBTENSOR_EXPR_SUITE_H
#define LIBTENSOR_LIBTENSOR_EXPR_SUITE_H

#include <libtest/test_suite.h>
#include "eval_plan_test.h"
#include "node_contract_test.h"
#include "node_transform_test.h"

using libtest::unit_test_factory;

namespace libtensor {


/** \defgroup libtensor_tests_expr Tests of the expression optimizer
    \ingroup libtensor_tests
 **/

/** \brief Test suite for the expression optimizer of libtensor

    This suite runs the following tests:
     - libtensor::eval_plan_test
     - libtensor::node_contract_test
     - libtensor::node_transform_test

    \ingroup libtensor_tests_expr
 **/
class libtensor_expr_suite : public libtest::test_suite {
private:
    unit_test_factory<eval_plan_test> m_utf_eval_plan;
    unit_test_factory<node_contract_test> m_utf_node_contract;
    unit_test_factory<node_transform_test> m_utf_node_transform;

public:
    //! Creates the suite
    libtensor_expr_suite();

};


} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_EXPR_SUITE_H
