#include <libtensor/block_sparse/batch_kernel_permute.h>
#include <libtensor/block_sparse/batch_kernel_contract2.h>
#include <libtensor/block_sparse/batch_kernel_add2.h>
#include <libtensor/block_sparse/sparse_btensor_new.h>
#include <libtensor/block_sparse/direct_sparse_btensor_new.h>
#include "batch_kernels_test.h"
#include "test_fixtures/permute_3d_sparse_120_test_f.h"
#include "test_fixtures/contract2_test_f.h"
#include "test_fixtures/subtract2_test_f.h"

using namespace std;

namespace libtensor {

void batch_kernels_test::perform() throw(libtest::test_exception) {
    test_batch_kernel_permute_A_direct();
    test_batch_kernel_contract2();
    test_batch_kernel_add2();
}

//A(i|j|k) = B(k|i|j)
void batch_kernels_test::test_batch_kernel_permute_A_direct() throw(libtest::test_exception)
{

    static const char *test_name = "batch_kernels_test::test_batch_kernel_permute_A_direct()";


    permute_3d_sparse_120_test_f tf;
    direct_sparse_btensor_new<3> A(tf.output_bispace);
    sparse_btensor_new<3> B(tf.input_bispace);


    letter i,j,k;
    idx_list perm_entries(1,1);
    perm_entries.push_back(2);
    perm_entries.push_back(0);
    batch_kernel_permute<double> bkp(A,B,perm_entries);

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
                    "batch_kernel_permute::generate_batch(...) did not produce correct result");
        }
    }
}

//C(i|l) = A(i|j|k) B(j|k|l)
void batch_kernels_test::test_batch_kernel_contract2() throw(libtest::test_exception)
{
    static const char *test_name = "batch_kernels_test::test_batch_kernel_contract2()";

    contract2_test_f tf;

    sparse_btensor_new<3> A(tf.spb_A,tf.A_arr,true);
    sparse_btensor_new<3> B(tf.spb_B,tf.B_arr,true);
    direct_sparse_btensor_new<2> C_direct(tf.spb_C);

    multimap<size_t,size_t> contr_map;
    contr_map.insert(idx_pair(1,3));
    contr_map.insert(idx_pair(2,4));
    batch_kernel_contract2<double> bkc2(C_direct,A,B,contr_map);

    //Check generating batch 1
    double C_arr[18];
    bispace_batch_map bbm;
    bbm[idx_pair(0,0)] = idx_pair(1,2);
    vector<double*> ptrs(1,C_arr);
    ptrs.push_back((double*)A.get_data_ptr());
    ptrs.push_back((double*)B.get_data_ptr());
    bkc2.init(ptrs,bbm);
    bkc2.generate_batch(ptrs,bbm);

    for(size_t i = 6 ; i < sizeof(tf.C_arr)/sizeof(tf.C_arr[0]); ++i)
    {
        if(C_arr[i - 6] != tf.C_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "batch_kernel_contract2::generate_batch(...) did not return correct batch 1");
        }
    }

    //Test grabbing entire array
    bkc2.init(ptrs,bispace_batch_map());
    bkc2.generate_batch(ptrs,bispace_batch_map());
    sparse_btensor_new<2> C(tf.spb_C,C_arr,true);
    sparse_btensor_new<2> C_correct(tf.spb_C,tf.C_arr,true);
    if(C != C_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_kernel_contract2::generate_batch(...) did not produce correct result");
    }
}

void batch_kernels_test::test_batch_kernel_add2() throw(libtest::test_exception)
{
    static const char *test_name = "batch_kernels_test::test_batch_kernel_add2()";

    subtract2_test_f tf;

    sparse_btensor_new<3> A(tf.spb_A,tf.A_arr,true);
    sparse_btensor_new<3> B(tf.spb_B,tf.B_arr,true);
    direct_sparse_btensor_new<3> C_direct(tf.spb_C);

    batch_kernel_add2<double> bka2(C_direct,A,B,1,-1);

    //Test grabbing entire array
    double C_arr[60];
    bispace_batch_map bbm;
    vector<double*> ptrs(1,C_arr);
    ptrs.push_back((double*)A.get_data_ptr());
    ptrs.push_back((double*)B.get_data_ptr());
    bka2.init(ptrs,bbm);
    bka2.generate_batch(ptrs,bbm);
    sparse_btensor_new<3> C(tf.spb_C,C_arr,true);
    sparse_btensor_new<3> C_correct(tf.spb_C,tf.C_arr,true);

    if(C != C_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "batch_kernel_contract2::generate_batch(...) did not produce correct result");
    }

    bbm[idx_pair(0,0)] = idx_pair(1,2);
    bka2.init(ptrs,bbm);
    bka2.generate_batch(ptrs,bbm);
    for(size_t i = 0; i < 40; ++i)
    {
        if(C_arr[i] != tf.C_arr[20+i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                "batch_kernel_contract2::generate_batch(...) did not produce correct result for batch 1");
        }
    } 

}

} // namespace libtensor
