#include <libtensor/block_sparse/batch_kernel_permute.h>
#include "batch_kernels_test.h"

using namespace std;

namespace libtensor {

void batch_kernels_test::perform() throw(libtest::test_exception) {
    test_batch_kernel_permute();
}


void batch_kernels_test::test_batch_kernel_permute() throw(libtest::test_exception)
{
    static const char *test_name = "batch_kernels_test::test_batch_kernel_permute()";
}

} // namespace libtensor
