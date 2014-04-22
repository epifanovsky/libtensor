#include <libtensor/block_sparse/batch_kernel_permute.h>
#include <libtensor/block_sparse/sparse_btensor.h>
#include <libtensor/block_sparse/direct_sparse_btensor.h>
#include "batch_kernels_test.h"
#include "test_fixtures/permute_3d_sparse_120_test_f.h"

using namespace std;

namespace libtensor {

void batch_kernels_test::perform() throw(libtest::test_exception) {
    /*test_batch_kernel_permute_A_direct();*/
}

//A(i|j|k) = B(k|i|j)
void batch_kernels_test::test_batch_kernel_permute_A_direct() throw(libtest::test_exception)
{

    static const char *test_name = "batch_kernels_test::test_batch_kernel_permute_A_direct()";


    permute_3d_sparse_120_test_f tf;
    direct_sparse_btensor<3> A(tf.output_bispace);
    sparse_btensor<3> B(tf.input_bispace);


    letter i,j,k;
    batch_kernel_permute<double> bkp(A(i|j|k),B(k|i|j));

    bispace_batch_map bbm;
    double output_batch_arr[20] = {0};
    bbm.insert(make_pair(idx_pair(0,1),idx_pair(0,1)));
    vector<double*> ptrs(1,output_batch_arr);
    ptrs.push_back(tf.input_arr);
    bkp.generate_batch(ptrs,bbm);

    double correct_output_batch_arr[20] = { //i = 0 j = 0 k = 0
                                            1,
                                            2,
                                            3,
                                            4,
                                            5,
                                            6,

                                            //i = 0 j = 0 k = 1
                                            21,27,
                                            22,28,
                                            23,29,
                                            24,30,
                                            25,31,
                                            26,32,

                                            //i = 1 j = 0 k = 0
                                            16,
                                            17};

    for(size_t i = 0;  i < 20; ++i)
    {
        if(output_batch_arr[i]  != correct_output_batch_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_permute_kernel::generate_batch(...) did not produce correct result");
        }
    }
}

} // namespace libtensor
