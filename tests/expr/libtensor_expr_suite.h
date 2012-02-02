#ifndef LIBTENSOR_LIBTENSOR_EXPR_SUITE_H
#define LIBTENSOR_LIBTENSOR_EXPR_SUITE_H

#include <libtest/test_suite.h>
#include "anytensor_test.h"
#include "bispace_test.h"
#include "bispace_expr_test.h"
#include "btensor_test.h"
#include "contract_test.h"
#include "diag_test.h"
#include "dirprod_test.h"
#include "dirsum_test.h"
#include "labeled_anytensor_test.h"
#include "letter_test.h"
#include "letter_expr_test.h"

using libtest::unit_test_factory;

namespace libtensor {


/** \defgroup libtensor_expr_tests Tests of the expressions interface
    \ingroup libtensor_tests
 **/


/** \brief Test suite for the expressions interface of libtensor

    This suite runs the following tests:
     - libtensor::anytensor_test
     - libtensor::bispace_test
     - libtensor::bispace_expr_test
     - libtensor::btensor_test
     - libtensor::contract_test
     - libtensor::diag_test
     - libtensor::dirprod_test
     - libtensor::dirsum_test
     - libtensor::labeled_anytensor_test
     - libtensor::letter_test
     - libtensor::letter_expr_test

    \ingroup libtensor_tests
 **/
class libtensor_expr_suite : public libtest::test_suite {
private:
    unit_test_factory<anytensor_test> m_utf_anytensor;
    unit_test_factory<bispace_test> m_utf_bispace;
    unit_test_factory<bispace_expr_test> m_utf_bispace_expr;
    unit_test_factory<btensor_test> m_utf_btensor;
    unit_test_factory<contract_test> m_utf_contract;
    unit_test_factory<diag_test> m_utf_diag;
    unit_test_factory<dirprod_test> m_utf_dirprod;
    unit_test_factory<dirsum_test> m_utf_dirsum;
    unit_test_factory<labeled_anytensor_test> m_utf_labeled_anytensor;
    unit_test_factory<letter_test> m_utf_letter;
    unit_test_factory<letter_expr_test> m_utf_letter_expr;

public:
    //! Creates the suite
    libtensor_expr_suite();

};


} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_EXPR_SUITE_H

