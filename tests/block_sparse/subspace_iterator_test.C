#include "subspace_iterator_test.h"

using namespace std;

namespace libtensor {

void subspace_iterator_test::perform() throw(libtest::test_exception) 
{
    test_get_block_index_dense();
}

void subspace_iterator_test::test_get_block_index_dense() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_iterator_test::test_get_block_index_dense()";
}

} // namespace libtensor
