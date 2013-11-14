#ifndef LIBTENSOR_LIBTENSOR_SPARSE_SUITE_H
#define LIBTENSOR_LIBTENSOR_SPARSE_SUITE_H

//TODO: REMOVE SPURIOUS HEADERS
#include <libtest/test_suite.h>
#include "sparse_block_tree_test.h"
#include "sparse_bispace_test.h"
#include "block_kernels_test.h"
#include "block_loop_test.h"
#include "sparse_btensor_test.h"
#include "direct_sparse_btensor_test.h"
#include "loop_list_sparsity_data_test.h"
#include "sparse_loop_list_test.h"

using libtest::unit_test_factory;

namespace libtensor {

/** \defgroup libtensor_tests_sparse Tests sparse tensor utilities 
    \brief Unit tests of the sparse capabilities of libtensor
    \ingroup libtensor_tests
 **/

/**
    \brief Test suite for the easy-to-use interface of libtensor
    \ingroup libtensor_tests

    This suite runs the following tests:
    \li libtensor::sparse_bispace_test
**/
class libtensor_sparse_suite : public libtest::test_suite {
private:
    unit_test_factory<block_kernels_test> m_utf_block_kernels;
    unit_test_factory<loop_list_sparsity_data_test> m_utf_loop_list_sparsity_data;
    unit_test_factory<block_loop_test> m_utf_block_loop;
    unit_test_factory<sparse_loop_list_test> m_utf_sparse_loop_list;
    unit_test_factory<sparse_block_tree_test> m_utf_sparse_block_tree;
    unit_test_factory<sparse_bispace_test> m_utf_sparse_bispace;
    unit_test_factory<sparse_btensor_test> m_utf_sparse_btensor;
    unit_test_factory<direct_sparse_btensor_test> m_utf_direct_sparse_btensor;
public:
    //! Creates the suite
    libtensor_sparse_suite();
};

} // namespace libtensor

#endif // LIBTENSOR_LIBTENSOR_SPARSE_SUITE_H

