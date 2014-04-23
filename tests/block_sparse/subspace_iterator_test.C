#include "subspace_iterator_test.h"
#include "test_fixtures/index_group_test_f.h"
#include <libtensor/block_sparse/subspace_iterator.h>

using namespace std;

namespace libtensor {

void subspace_iterator_test::perform() throw(libtest::test_exception) 
{
    test_get_block_index_dense();
}

void subspace_iterator_test::test_get_block_index_dense() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_iterator_test::test_get_block_index_dense()";

    index_groups_test_f tf = index_groups_test_f();
    subspace_iterator si(tf.bispace,0);

    if(si.get_block_index() != 0)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace_iterator::get_block_index(...) did not return correct value for subspace 0 (dense subspace)");
    }
}

} // namespace libtensor
