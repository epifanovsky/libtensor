#ifndef LIBTENSOR_LIBTENSOR_CTF_BLOCK_TENSOR_SUITE_H
#define LIBTENSOR_LIBTENSOR_CTF_BLOCK_TENSOR_SUITE_H

#include <libtest/test_suite.h>
#include "ctf_btod_collect_test.h"
#include "ctf_btod_contract2_test.h"
#include "ctf_btod_copy_test.h"
#include "ctf_btod_diag_test.h"
#include "ctf_btod_set_test.h"

using libtest::unit_test_factory;

namespace libtensor {


/** \defgroup libtensor_ctf_block_tensor_tests Tests for CTF block tensors and
        operations
    \ingroup libtensor_tests
 **/


/** \brief Test suite for CTF block tensors

    This suite runs the following tests:
     - libtensor::ctf_btod_collect_test
     - libtensor::ctf_btod_contract2_test
     - libtensor::ctf_btod_copy_test
     - libtensor::ctf_btod_diag_test
     - libtensor::ctf_btod_set_test

    \ingroup libtensor_ctf_block_tensor_tests
 **/
class libtensor_ctf_block_tensor_suite : public libtest::test_suite {
private:
    unit_test_factory<ctf_btod_collect_test> m_utf_ctf_btod_collect;
    unit_test_factory<ctf_btod_contract2_test> m_utf_ctf_btod_contract2;
    unit_test_factory<ctf_btod_copy_test> m_utf_ctf_btod_copy;
    unit_test_factory<ctf_btod_diag_test> m_utf_ctf_btod_diag;
    unit_test_factory<ctf_btod_set_test> m_utf_ctf_btod_set;

public:
    //! Creates the suite
    libtensor_ctf_block_tensor_suite();

};


} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_CTF_BLOCK_TENSOR_SUITE_H

