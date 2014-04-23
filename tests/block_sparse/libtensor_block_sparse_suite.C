#include "libtensor_block_sparse_suite.h"

namespace libtensor {

libtensor_sparse_suite::libtensor_sparse_suite() :
    libtest::test_suite("libtensor") {

    add_test("block_kernels", m_utf_block_kernels);
    add_test("block_loop", m_utf_block_loop);
    add_test("sparse_loop_list", m_utf_sparse_loop_list);
    add_test("sparsity_fuser", m_utf_sparsity_fuser);
    add_test("sparse_block_tree_iterator", m_utf_sparse_block_tree_iterator);
    add_test("sparse_block_tree", m_utf_sparse_block_tree);
    add_test("sparse_bispace", m_utf_sparse_bispace);
    add_test("sparse_btensor", m_utf_sparse_btensor);
    add_test("direct_sparse_btensor", m_utf_direct_sparse_btensor);
    add_test("sparse_loop_grouper", m_utf_sparse_loop_grouper);
    add_test("blas_isomorphism", m_utf_blas_isomorphism);
    add_test("batch_kernels", m_utf_batch_kernels);
    add_test("batch_list_builder", m_utf_batch_list_builder);
    add_test("subspace_iterator", m_utf_subspace_iterator);
}

}

