#include "batch_provider_test.h"
#include "test_fixtures/permute_3d_sparse_120_test_f.h"
#include <libtensor/block_sparse/sparse_btensor.h>

using namespace std;

namespace libtensor {

void batch_provider_test::perform() throw(libtest::test_exception) 
{
    test_permute_3d_sparse_120();
}

void batch_provider_test::test_permute_3d_sparse_120() throw(libtest::test_exception)
{
    static const char *test_name = "batch_provider_test::test_permute_3d_sparse_120()";

    /*if(si.get_block_index() != 0)*/
    /*{*/
        /*fail_test(test_name,__FILE__,__LINE__,*/
                /*"subspace_iterator::get_block_index(...) did not return correct value for subspace 0 (dense subspace)");*/
    /*}*/
}

} // namespace libtensor
