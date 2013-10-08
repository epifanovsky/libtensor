#include "libtensor_block_sparse_suite.h"

namespace libtensor {

libtensor_sparse_suite::libtensor_sparse_suite() :
    libtest::test_suite("libtensor") {

    add_test("sparse_bispace", m_utf_sparse_bispace);
    add_test("sparse_btensor", m_utf_sparse_btensor);
    add_test("block_kernels", m_utf_block_kernels);
    add_test("block_loop", m_utf_block_loop);
}

}

