#include <libtensor/block_sparse/batch_kernel_permute.h>
#include "batch_kernels_test.h"
#include "test_fixtures/permute_3d_sparse_120_test_f.h"

using namespace std;

namespace libtensor {

void batch_kernels_test::perform() throw(libtest::test_exception) {
    test_batch_kernel_permute();
}

//A(i|j|k) = B(k|i|j)
void batch_kernels_test::test_batch_kernel_permute() throw(libtest::test_exception)
{

    static const char *test_name = "batch_kernels_test::test_batch_kernel_permute()";

#if 0
const double permute_3d_sparse_120_test_f::s_input_arr[35] = { //k = 0, i = 0; j = 0
                         1,2,
                         3,4,
                         5,6,

                         //k = 0, i = 1, j = 0
                         16,17,

                         //k = 1, i = 0, j = 0
                         21,22,
                         23,24,
                         25,26,
                         27,28,
                         29,30,
                         31,32,

    permute_3d_sparse_120_test_f tf;
    sparse_btensor<3> A(tf.output_bispace);
    sparse_btensor<3> B(tf.input_bispace);

    vector<double*> ptrs(1,);

    letter i,j,k;
    batch_kernel_permute bkp(output_btensor(i|j|k),input_btensor(k|i|j));

    bispace_batch_map bbm;
    bbm.insert(idx_pair(0,1),idx_pair(0,1));
    bkp.generate_batch(bbm,ptrs);
#endif
}

} // namespace libtensor
