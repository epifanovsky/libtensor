#ifndef LIBTENSOR_LIBTENSOR_CTF_DENSE_TENSOR_SUITE_H
#define LIBTENSOR_LIBTENSOR_CTF_DENSE_TENSOR_SUITE_H

#include <libtest/test_suite.h>
#include "ctf_dense_tensor_test.h"
#include "ctf_tod_distribute_test.h"

using libtest::unit_test_factory;

namespace libtensor {


/** \defgroup libtensor_ctf_dense_tensor_tests Tests for CTF tensors and
        operations
    \ingroup libtensor_tests
 **/


/** \brief Test suite for CTF tensors

    This suite runs the following tests:
     - libtensor::ctf_dense_tensor_test
     - libtensor::ctf_tod_distribute_test

    \ingroup libtensor_ctf_dense_tensor_tests
 **/
class libtensor_ctf_dense_tensor_suite : public libtest::test_suite {
private:
    unit_test_factory<ctf_dense_tensor_test> m_utf_ctf_dense_tensor;
    unit_test_factory<ctf_tod_distribute_test> m_utf_ctf_tod_distribute;

public:
    //! Creates the suite
    libtensor_ctf_dense_tensor_suite();

};


} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_CTF_DENSE_TENSOR_SUITE_H

