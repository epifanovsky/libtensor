#include "libtensor_block_sparse_suite.h"

namespace libtensor {

libtensor_sparse_suite::libtensor_sparse_suite() :
    libtest::test_suite("libtensor") {

    add_test("block_kernels", m_utf_block_kernels);
    add_test("loop_list_sparsity_data", m_utf_loop_list_sparsity_data);
    add_test("block_loop", m_utf_block_loop);
    add_test("sparse_loop_list", m_utf_sparse_loop_list);
    add_test("sparse_loop_iterator", m_utf_sparse_loop_iterator);
    add_test("sparse_block_tree_iterator", m_utf_sparse_block_tree_iterator);
    add_test("sparse_block_tree", m_utf_sparse_block_tree);
    add_test("sparse_bispace", m_utf_sparse_bispace);
    add_test("sparse_btensor", m_utf_sparse_btensor);
    add_test("direct_sparse_btensor", m_utf_direct_sparse_btensor);
}

}

