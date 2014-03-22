#ifndef LIBTENSOR_LIBTENSOR_CTF_IFACE_SUITE_H
#define LIBTENSOR_LIBTENSOR_CTF_IFACE_SUITE_H

#include <libtest/test_suite.h>
#include "ctf_btensor_test.h"
#include "ctf_expr_test.h"

using libtest::unit_test_factory;

namespace libtensor {


/** \defgroup libtensor_ctf_iface_tests Tests for CTF expressions interface
    \ingroup libtensor_tests
 **/


/** \brief Test suite for the expressions interface using CTF tensors

    This suite runs the following tests:
     - libtensor::ctf_btensor_test
     - libtensor::ctf_expr_test

    \ingroup libtensor_ctf_iface_tests
 **/
class libtensor_ctf_iface_suite : public libtest::test_suite {
private:
    unit_test_factory<ctf_btensor_test> m_utf_ctf_btensor;
    unit_test_factory<ctf_expr_test> m_utf_ctf_expr;

public:
    //! Creates the suite
    libtensor_ctf_iface_suite();

};


} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_CTF_IFACE_SUITE_H

