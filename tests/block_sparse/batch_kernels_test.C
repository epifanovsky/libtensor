#include <libtensor/block_sparse/batch_kernel_permute.h>
#include <libtensor/block_sparse/batch_kernel_contract2.h>
#include <libtensor/block_sparse/batch_kernel_add2.h>
#include <libtensor/block_sparse/batch_kernel_unblock.h>
#include <libtensor/block_sparse/sparse_btensor.h>
#include <libtensor/block_sparse/direct_sparse_btensor.h>
#include "batch_kernels_test.h"
#include "test_fixtures/permute_3d_sparse_120_test_f.h"
#include "test_fixtures/contract2_test_f.h"
#include "test_fixtures/contract2_dense_dense_test_f.h"
#include "test_fixtures/subtract2_test_f.h"

using namespace std;

namespace libtensor {

void batch_kernels_test::perform() throw(libtest::test_exception) {
    test_batch_kernel_permute_A_direct();
    test_batch_kernel_contract2();
    test_batch_kernel_add2();
    test_batch_kernel_unblock();
    test_batch_kernel_unblock_direct();
}

//A(i|j|k) = B(k|i|j)
void batch_kernels_test::test_batch_kernel_permute_A_direct() throw(libtest::test_exception)
{

    static const char *test_name = "batch_kernels_test::test_batch_kernel_permute_A_direct()";


    permute_3d_sparse_120_test_f tf;
    direct_sparse_btensor<3> A(tf.output_bispace);
    sparse_btensor<3> B(tf.input_bispace);


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

    sparse_btensor<3> A(tf.spb_A,tf.A_arr,true);
    sparse_btensor<3> B(tf.spb_B,tf.B_arr,true);
    direct_sparse_btensor<2> C_direct(tf.spb_C);

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
    sparse_btensor<2> C(tf.spb_C,C_arr,true);
    sparse_btensor<2> C_correct(tf.spb_C,tf.C_arr,true);
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

    sparse_btensor<3> A(tf.spb_A,tf.A_arr,true);
    sparse_btensor<3> B(tf.spb_B,tf.B_arr,true);
    direct_sparse_btensor<3> C_direct(tf.spb_C);

    batch_kernel_add2<double> bka2(C_direct,A,B,1,-1);

    //Test grabbing entire array
    double C_arr[60];
    bispace_batch_map bbm;
    vector<double*> ptrs(1,C_arr);
    ptrs.push_back((double*)A.get_data_ptr());
    ptrs.push_back((double*)B.get_data_ptr());
    bka2.init(ptrs,bbm);
    bka2.generate_batch(ptrs,bbm);
    sparse_btensor<3> C(tf.spb_C,C_arr,true);
    sparse_btensor<3> C_correct(tf.spb_C,tf.C_arr,true);

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

void batch_kernels_test::test_batch_kernel_unblock() throw(libtest::test_exception)
{
    static const char *test_name = "batch_kernels_test::test_batch_kernel_unblock()";

    contract2_dense_dense_test_f tf;

    double correct_A_unblocked_arr_0[60] = {//i = 0 j = 0 k = 0 (1,2,2)
                                            1,2,
                                            3,4,

                                            //i = 0 j = 0 k = 1 (1,2,3)
                                            5,6,7,
                                            8,9,10,

                                            //i = 0 j = 1 k = 0 (1,2,2)
                                            11,12,
                                            13,14,

                                            //i = 0 j = 1 k = 1 (1,2,3)
                                            15,16,17,
                                            18,19,20,

                                            //i = 1 j = 0 k = 0 (1,2,2)
                                            21,22,
                                            23,24,


                                            //i = 1 j = 0 k = 1 (1,2,3)
                                            29,30,31,
                                            32,33,34,

                                            //i = 1 j = 1 k = 0 (1,2,2)
                                            41,42,
                                            43,44,

                                            //i = 1 j = 1 k = 1 (1,2,3)
                                            49,50,51,
                                            52,53,54,

                                            //i = 2 j = 0 k = 0 (1,2,2)
                                            25,26,
                                            27,28,

                                            //i = 2 j = 0 k = 1 (1,2,3)
                                            35,36,37,
                                            38,39,40,

                                            //i = 2 j = 1 k = 0 (1,2,2)
                                            45,46,
                                            47,48,

                                            //i = 2 j = 1 k = 1 (1,2,3)
                                            55,56,57,
                                            58,59,60};

    double A_unblocked_arr_0[60] = {0};
    vector<double*> ptrs(1,A_unblocked_arr_0);
    ptrs.push_back(tf.A_arr);
    bispace_batch_map bbm;
    batch_kernel_unblock<double> k_un_0(tf.spb_A,0);
    k_un_0.generate_batch(ptrs,bbm);
    for(size_t i = 0; i < sizeof(correct_A_unblocked_arr_0)/sizeof(correct_A_unblocked_arr_0[0]); ++i)
    {
        if(A_unblocked_arr_0[i] != correct_A_unblocked_arr_0[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                "batch_kernel_unblock::generate_batch(...) did not produce correct result for A unbatched subspace 0");
        }
    }

    double correct_A_unblocked_arr_1[60] = {//i = 0 j = 0 k = 0
                                            1,2,

                                            //i = 0 j = 0 k = 1
                                            5,6,7,

                                            //i = 0 j = 1 k = 0
                                            3,4,

                                            //i = 0 j = 1 k = 1
                                            8,9,10,

                                            //i = 0 j = 2 k = 0
                                            11,12,

                                            //i = 0 j = 2 k = 1
                                            15,16,17,

                                            //i = 0 j = 3 k = 0
                                            13,14,

                                            //i = 0 j = 3 k = 1
                                            18,19,20,

                                            //i = 1 j = 0 k = 0
                                            21,22,
                                            //i = 1 j = 0 k = 1
                                            29,30,31,
                                            //i = 1 j = 1 k = 0
                                            23,24,
                                            //i = 1 j = 1 k = 1
                                            32,33,34,
                                            //i = 1 j = 2 k = 0
                                            41,42,
                                            //i = 1 j = 2 k = 1
                                            49,50,51,
                                            //i = 1 j = 3 k = 0
                                            43,44,
                                            //i = 1 j = 3 k = 1
                                            52,53,54,

                                            //i = 1 j = 0 k = 0
                                            25,26,
                                            //i = 1 j = 0 k = 1
                                            35,36,37,
                                            //i = 1 j = 1 k = 0
                                            27,28,
                                            //i = 1 j = 1 k = 1
                                            38,39,40,
                                            //i = 1 j = 2 k = 0
                                            45,46,
                                            //i = 1 j = 2 k = 1
                                            55,56,57,
                                            //i = 1 j = 3 k = 0
                                            47,48,
                                            //i = 1 j = 3 k = 1
                                            58,59,60};

    double A_unblocked_arr_1[60] = {0};
    ptrs[0] = A_unblocked_arr_1;
    batch_kernel_unblock<double> k_un_1(tf.spb_A,1);
    k_un_1.generate_batch(ptrs,bbm);
    for(size_t i = 0; i < sizeof(correct_A_unblocked_arr_1)/sizeof(correct_A_unblocked_arr_1[0]); ++i)
    {
        if(A_unblocked_arr_1[i] != correct_A_unblocked_arr_1[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                "batch_kernel_unblock::generate_batch(...) did not produce correct result for A unbatched subspace 1");
        }
    }

    double correct_A_unblocked_arr_2[60] = {//i = 0 j = 0 k
                                            1,2,5,6,7,
                                            3,4,8,9,10,

                                            //i = 0 j = 1 k
                                            11,12,15,16,17,
                                            13,14,18,19,20,

                                            //i = 1 j = 0 k
                                            21,22,29,30,31,
                                            23,24,32,33,34,
                                            25,26,35,36,37,
                                            27,28,38,39,40,

                                            //i = 1 j = 1 k
                                            41,42,49,50,51,
                                            43,44,52,53,54,
                                            45,46,55,56,57,
                                            47,48,58,59,60};
    double A_unblocked_arr_2[60] = {0};
    ptrs[0] = A_unblocked_arr_2;
    batch_kernel_unblock<double> k_un_2(tf.spb_A,2);
    k_un_2.generate_batch(ptrs,bbm);
    for(size_t i = 0; i < sizeof(correct_A_unblocked_arr_2)/sizeof(correct_A_unblocked_arr_2[0]); ++i)
    {
        if(A_unblocked_arr_2[i] != correct_A_unblocked_arr_2[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                "batch_kernel_unblock::generate_batch(...) did not produce correct result for A unbatched subspace 2");
        }
    }


}

void batch_kernels_test::test_batch_kernel_unblock_direct() throw(libtest::test_exception)
{
    static const char *test_name = "batch_kernels_test::test_batch_kernel_unblock_direct()";

    contract2_dense_dense_test_f tf;

    double correct_A_unblocked_arr_0_1[40] = {//i = 1 j = 0 k = 0 (1,2,2)
                                              21,22,
                                              23,24,

                                              //i = 1 j = 0 k = 1 (1,2,3)
                                              29,30,31,
                                              32,33,34,

                                              //i = 1 j = 1 k = 0 (1,2,2)
                                              41,42,
                                              43,44,

                                              //i = 1 j = 1 k = 1 (1,2,3)
                                              49,50,51,
                                              52,53,54,

                                              //i = 2 j = 0 k = 0 (1,2,2)
                                              25,26,
                                              27,28,

                                              //i = 2 j = 0 k = 1 (1,2,3)
                                              35,36,37,
                                              38,39,40,

                                              //i = 2 j = 1 k = 0 (1,2,2)
                                              45,46,
                                              47,48,

                                              //i = 2 j = 1 k = 1 (1,2,3)
                                              55,56,57,
                                              58,59,60};

    double A_unblocked_arr_0_1[40] = {0};
    vector<double*> ptrs(1,A_unblocked_arr_0_1);
    ptrs.push_back(tf.A_arr);
    bispace_batch_map bbm_0;
    bbm_0.insert(make_pair(idx_pair(0,0),idx_pair(1,2)));
    bbm_0.insert(make_pair(idx_pair(1,0),idx_pair(1,2)));
    batch_kernel_unblock<double> k_un_0(tf.spb_A,0);
    k_un_0.generate_batch(ptrs,bbm_0);
    for(size_t i = 0; i < sizeof(correct_A_unblocked_arr_0_1)/sizeof(correct_A_unblocked_arr_0_1[0]); ++i)
    {
        if(A_unblocked_arr_0_1[i] != correct_A_unblocked_arr_0_1[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                "batch_kernel_unblock::generate_batch(...) did not produce correct result for A subspace 0 batch 1");
        }
    }

    double correct_A_unblocked_arr_1_0[30] = {//i = 0 j = 0 k = 0
                                              1,2,

                                              //i = 0 j = 0 k = 1
                                              5,6,7,

                                              //i = 0 j = 1 k = 0
                                              3,4,

                                              //i = 0 j = 1 k = 1
                                              8,9,10,

                                              //i = 1 j = 0 k = 0
                                              21,22,
                                              //i = 1 j = 0 k = 1
                                              29,30,31,
                                              //i = 1 j = 1 k = 0
                                              23,24,
                                              //i = 1 j = 1 k = 1
                                              32,33,34,
                                              //i = 1 j = 0 k = 0
                                              25,26,
                                              //i = 1 j = 0 k = 1
                                              35,36,37,
                                              //i = 1 j = 1 k = 0
                                              27,28,
                                              //i = 1 j = 1 k = 1
                                              38,39,40};
    double A_unblocked_arr_1_0[30] = {0};
    ptrs[0] = A_unblocked_arr_1_0;
    bispace_batch_map bbm_1;
    bbm_1.insert(make_pair(idx_pair(0,1),idx_pair(0,1)));
    bbm_1.insert(make_pair(idx_pair(1,1),idx_pair(0,1)));
    batch_kernel_unblock<double> k_un_1(tf.spb_A,1);
    k_un_1.generate_batch(ptrs,bbm_1);
    for(size_t i = 0; i < sizeof(correct_A_unblocked_arr_1_0)/sizeof(correct_A_unblocked_arr_1_0[0]); ++i)
    {
        if(A_unblocked_arr_1_0[i] != correct_A_unblocked_arr_1_0[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                "batch_kernel_unblock::generate_batch(...) did not produce correct result for A subspace 1 batch 0");
        }
    }
}

} // namespace libtensor
