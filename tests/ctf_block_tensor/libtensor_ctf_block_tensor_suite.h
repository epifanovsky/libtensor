#ifndef LIBTENSOR_LIBTENSOR_CTF_BLOCK_TENSOR_SUITE_H
#define LIBTENSOR_LIBTENSOR_CTF_BLOCK_TENSOR_SUITE_H

#include <libtest/test_suite.h>
#include "ctf_btod_collect_test.h"
#include "ctf_btod_contract2_test.h"
#include "ctf_btod_copy_test.h"
#include "ctf_btod_diag_test.h"
#include "ctf_btod_dirsum_test.h"
#include "ctf_btod_distribute_test.h"
#include "ctf_btod_dotprod_test.h"
#include "ctf_btod_ewmult2_test.h"
#include "ctf_btod_mult_test.h"
#include "ctf_btod_mult1_test.h"
#include "ctf_btod_random_test.h"
#include "ctf_btod_scale_test.h"
#include "ctf_btod_set_test.h"
#include "ctf_btod_symmetrize2_test.h"
#include "ctf_btod_symmetrize3_test.h"
#include "ctf_btod_trace_test.h"
#include "ctf_symmetry_builder_test.h"

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
     - libtensor::ctf_btod_dirsum_test
     - libtensor::ctf_btod_distribute_test
     - libtensor::ctf_btod_dotprod_test
     - libtensor::ctf_btod_ewmult2_test
     - libtensor::ctf_btod_mult_test
     - libtensor::ctf_btod_mult1_test
     - libtensor::ctf_btod_random_test
     - libtensor::ctf_btod_scale_test
     - libtensor::ctf_btod_set_test
     - libtensor::ctf_btod_symmetrize2_test
     - libtensor::ctf_btod_symmetrize3_test
     - libtensor::ctf_btod_trace_test
     - libtensor::ctf_symmetry_builder_test

    \ingroup libtensor_ctf_block_tensor_tests
 **/
class libtensor_ctf_block_tensor_suite : public libtest::test_suite {
private:
    unit_test_factory<ctf_btod_collect_test> m_utf_ctf_btod_collect;
    unit_test_factory<ctf_btod_contract2_test> m_utf_ctf_btod_contract2;
    unit_test_factory<ctf_btod_copy_test> m_utf_ctf_btod_copy;
    unit_test_factory<ctf_btod_diag_test> m_utf_ctf_btod_diag;
    unit_test_factory<ctf_btod_dirsum_test> m_utf_ctf_btod_dirsum;
    unit_test_factory<ctf_btod_distribute_test> m_utf_ctf_btod_distribute;
    unit_test_factory<ctf_btod_dotprod_test> m_utf_ctf_btod_dotprod;
    unit_test_factory<ctf_btod_ewmult2_test> m_utf_ctf_btod_ewmult2;
    unit_test_factory<ctf_btod_mult_test> m_utf_ctf_btod_mult;
    unit_test_factory<ctf_btod_mult1_test> m_utf_ctf_btod_mult1;
    unit_test_factory<ctf_btod_random_test> m_utf_ctf_btod_random;
    unit_test_factory<ctf_btod_scale_test> m_utf_ctf_btod_scale;
    unit_test_factory<ctf_btod_set_test> m_utf_ctf_btod_set;
    unit_test_factory<ctf_btod_symmetrize2_test> m_utf_ctf_btod_symmetrize2;
    unit_test_factory<ctf_btod_symmetrize3_test> m_utf_ctf_btod_symmetrize3;
    unit_test_factory<ctf_btod_trace_test> m_utf_ctf_btod_trace;
    unit_test_factory<ctf_symmetry_builder_test> m_utf_ctf_symmetry_builder;

public:
    //! Creates the suite
    libtensor_ctf_block_tensor_suite();

};


} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_CTF_BLOCK_TENSOR_SUITE_H

