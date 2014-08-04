#include <libtensor/block_sparse/direct_sparse_btensor.h>
#include <libtensor/block_sparse/sparse_btensor.h>
#include <libtensor/block_sparse/direct_sparse_btensor_new.h>
#include <libtensor/block_sparse/sparse_btensor_new.h>
#include <libtensor/expr/operators/contract.h>
#include "direct_sparse_btensor_test.h"
#include "test_fixtures/contract2_test_f.h"
#include "test_fixtures/contract2_subtract2_nested_test_f.h"
#include <math.h>

namespace libtensor {

void direct_sparse_btensor_test::perform() throw(libtest::test_exception) {
    test_contract2_direct_rhs();
    test_contract2_subtract2_nested();
    test_custom_batch_provider();
    /*test_force_batch_index();*/
}

//TODO: group with other test in test fixture
void direct_sparse_btensor_test::test_contract2_direct_rhs() throw(libtest::test_exception)
{
    static const char *test_name = "direct_sparse_btensor_test::test_contract2_direct_rhs()";
    //Make batch memory just big enough to fit i = 1 batch of C 
    //in addition to existing tensors held in core
    //This will force partitioning into i = 0 and i = 1
    memory_reserve mr_0(360+480+144+168+96);
    memory_reserve mr_1(360+480+144+168+96-1);
    contract2_test_f tf;

    /*** FIRST STEP - SET UP DIRECT TENSOR ***/
    sparse_btensor_new<3> A(tf.spb_A,tf.A_arr,true);
    sparse_btensor_new<3> B(tf.spb_B,tf.B_arr,true);
    direct_sparse_btensor_new<2> C(tf.spb_C);

    letter i,j,k,l;
    C(i|l) = contract(j|k,A(i|j|k),B(j|k|l));

    /*** SECOND STEP - USE DIRECT TENSOR ***/

    //(ml) sparsity
    size_t seq_0_arr_ml[2] = {0,1};
    size_t seq_1_arr_ml[2] = {0,2};
    size_t seq_2_arr_ml[2] = {1,0};
    size_t seq_3_arr_ml[2] = {1,2};

    std::vector< sequence<2,size_t> > ml_sig_blocks(4);
    for(size_t i = 0; i < 2; ++i) ml_sig_blocks[0][i] = seq_0_arr_ml[i];
    for(size_t i = 0; i < 2; ++i) ml_sig_blocks[1][i] = seq_1_arr_ml[i];
    for(size_t i = 0; i < 2; ++i) ml_sig_blocks[2][i] = seq_2_arr_ml[i];
    for(size_t i = 0; i < 2; ++i) ml_sig_blocks[3][i] = seq_3_arr_ml[i];

    //Bispace for m
    sparse_bispace<1> spb_m(6);
    std::vector<size_t> split_points_m;
    split_points_m.push_back(3);
    spb_m.split(split_points_m);


    sparse_bispace<2> spb_D = spb_m % tf.spb_l << ml_sig_blocks;
    double D_arr[21] = {  //m = 0 l = 1
                          1,2,3,
                          4,5,6,
                          7,8,9,

                          //m = 0 l = 2
                          10,
                          11,
                          12,

                          //m = 1 l = 0
                          13,14,
                          15,16,
                          17,18,

                          //m = 1 l = 2
                          19,
                          20,
                          21
                          };

    sparse_btensor_new<2> D(spb_D,D_arr,true);

    sparse_bispace<2> spb_E = spb_m | tf.spb_i;
    sparse_btensor_new<2> E(spb_E);
    letter m;

    A.set_memory_reserve(mr_0);
    B.set_memory_reserve(mr_0);
    D.set_memory_reserve(mr_0);
    E.set_memory_reserve(mr_0);
    E(m|i) = contract(l,D(m|l),C(i|l));

    double E_correct_arr[18] = { //m = 0 i = 0
                                 22012,
                                 47279,
                                 72546,

                                 //m = 0 i = 1
                                 122608,133324,
                                 240894,261085,
                                 359180,388846,

                                 //m = 1 i = 0
                                 55327,
                                 62548,
                                 69769,

                                 //m = 1 i = 1
                                 302748,330026,
                                 339191,369663,
                                 375634,409300
                               };


    sparse_btensor_new<2> E_correct(spb_E,E_correct_arr,true);
    if(E != E_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "contract(...) did not produce correct result");
    }

    //Now make sure we run out of memory with 1 less byte than minimal necessary
    A.set_memory_reserve(mr_1);
    B.set_memory_reserve(mr_1);
    D.set_memory_reserve(mr_1);
    E.set_memory_reserve(mr_1);

    bool threw_exception = false;
    try
    {
        E(m|i) = contract(l,D(m|l),C(i|l));
    }
    catch(out_of_memory&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "contract(...) did not fail with insufficient memory");
    }
}

void direct_sparse_btensor_test::test_contract2_subtract2_nested() throw(libtest::test_exception)
{
    static const char *test_name = "direct_sparse_btensor_test::test_contract2_subtract2_nested()";
    //Make batch memory just big enough to fit i = 1 batch of G and C simultaneously
    //This will force partitioning into i = 0 and i = 1
    memory_reserve mr_0(360+480+144+168+144+2*96); 
    memory_reserve mr_1(360+480+144+168+144+2*96-1); 

    contract2_subtract2_nested_test_f tf;
    tf.A.set_memory_reserve(mr_0);
    tf.B.set_memory_reserve(mr_0);
    tf.F.set_memory_reserve(mr_0);
    tf.D.set_memory_reserve(mr_0);
    tf.E.set_memory_reserve(mr_0);

    letter i,j,k,l;
    tf.C(i|l) = contract(j|k,tf.A(i|j|k),tf.B(j|k|l));
    tf.G(i|l) = tf.C(i|l) - tf.F(i|l);
    letter m;
    tf.E(m|i) = contract(l,tf.D(m|l),tf.G(i|l));

    if(tf.E != tf.E_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "contract(...) did not produce correct result");
    }

    tf.A.set_memory_reserve(mr_1);
    tf.B.set_memory_reserve(mr_1);
    tf.F.set_memory_reserve(mr_1);
    tf.D.set_memory_reserve(mr_1);
    tf.E.set_memory_reserve(mr_1);

    bool threw_exception = false;
    try
    {
        tf.E(m|i) = contract(l,tf.D(m|l),tf.G(i|l));
    }
    catch(out_of_memory&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "out_of_memory not thrown when not enough memory given");
    }
}

//We simulate a custom batch provider, as might occur if we were dynamically generating elements of the tensor through
//some numerical method 
void direct_sparse_btensor_test::test_custom_batch_provider() throw(libtest::test_exception)
{
    static const char *test_name = "direct_sparse_btensor_test::test_custom_batch_provider()";

    sparse_bispace<1> spb_i(5);
    std::vector<size_t> split_points_i;
    split_points_i.push_back(2);
    spb_i.split(split_points_i);

    sparse_bispace<1> spb_j(6);
    std::vector<size_t> split_points_j;
    split_points_j.push_back(3);
    spb_j.split(split_points_j);


    //Our contrived batcher computes a matrix where each term is the product of a power of 2 (row) and a fibonacci
    //number (col) 
    class two_n_fibonnaci_batch_provider : public batch_provider_i<double>
    {
    public:
        sparse_bispace<1> m_spb_i;
        sparse_bispace<1> m_spb_j;
        size_t fibonnaci(size_t n)
        { 
            if(n == 0) { return 0; }
            else if(n == 1) { return 1; }
            else { return fibonnaci(n - 1) + fibonnaci(n - 2); }
        }

        virtual void get_batch(double* output_ptr,const bispace_batch_map& bbm = bispace_batch_map()) 
        {
            idx_pair batch = bbm.begin()->second; 
            size_t batch_off = 0;
            for(size_t i_block_idx = batch.first; i_block_idx < batch.second; ++i_block_idx)
            {
                size_t i_block_off = m_spb_i.get_block_abs_index(i_block_idx);
                size_t i_block_sz = m_spb_i.get_block_size(i_block_idx);
                for(size_t j_block_idx = 0; j_block_idx < m_spb_j.get_n_blocks(); ++j_block_idx)
                {
                    size_t j_block_off = m_spb_j.get_block_abs_index(j_block_idx);
                    size_t j_block_sz = m_spb_j.get_block_size(j_block_idx);
                    for(size_t i_element_idx = 0; i_element_idx < i_block_sz; ++i_element_idx)
                    {
                        for(size_t j_element_idx = 0; j_element_idx < j_block_sz; ++j_element_idx)
                        {
                            size_t two_n = (size_t) pow(2,i_block_off+i_element_idx);
                            size_t fibonnaci_n = fibonnaci(j_block_off+j_element_idx);
                            output_ptr[batch_off] = two_n*fibonnaci_n;
                            ++batch_off;
                        }
                    }
                }
            }
        }
    public:
        two_n_fibonnaci_batch_provider(const sparse_bispace<1>& spb_i,const sparse_bispace<1>& spb_j) : m_spb_i(spb_i),m_spb_j(spb_j) {}
    };

    std::vector<sparse_bispace_any_order> bispaces(1,spb_i|spb_j);

    memory_reserve mr_0(240+144);
    memory_reserve mr_1(240+144-1);
    direct_sparse_btensor_new<2> A(spb_i|spb_j);
    two_n_fibonnaci_batch_provider bp(spb_i,spb_j);
    A.set_batch_provider(bp);
    sparse_btensor_new<2> B(spb_i|spb_j);
    letter i,j;
    B.set_memory_reserve(mr_0);
    B(i|j) = A(i|j);

    double B_correct_arr[30] = { //i = 0 j = 0
                                 0,1,1,
                                 0,2,2,

                                 //i = 0 j = 1
                                 2,3,5,
                                 4,6,10,

                                 //i = 1 j = 0
                                 0,4,4,
                                 0,8,8,
                                 0,16,16,

                                 //i = 1 j = 1
                                 8,12,20,
                                 16,24,40,
                                 32,48,80};
                                        

    sparse_btensor_new<2> B_correct(spb_i|spb_j,B_correct_arr,true);
    if(B != B_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "Custom batch provider returned incorrect tensor");
    }

    B.set_memory_reserve(mr_1);
    bool threw_exception = false;
    try
    {
        B(i|j) = A(i|j);
    }
    catch(out_of_memory&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "out_of_memory not thrown when not enough memory given");
    }
}

#if 0
//Due to constraints of external code that may be providing us with some batches
//we may need to force the choice of a particular index to batch over.
void direct_sparse_btensor_test::test_force_batch_index() throw(libtest::test_exception)
{
    static const char *test_name = "direct_sparse_btensor_test::test_force_batch_index()";

    //Block major
    double A_arr[26] = { //i = 0 j = 1 k = 2
                         1,2,
                         3,4,

                         //i = 1 j = 0 k = 1
                         5,
                         6,

                         //i = 1 j = 1 k = 1
                         7,
                         8,
                         9,
                         10,

                         //i = 2 j = 0 k = 0
                         11,12,
                         13,14,

                         //i = 2 j = 1 k = 1
                         15,
                         16,
                         17,
                         18,

                         //i = 2  j = 1 k = 2
                         19,20,
                         21,22,
                         23,24,
                         25,26};

    //Bispace for i 
    sparse_bispace<1> spb_i(5);
    std::vector<size_t> split_points_i;
    split_points_i.push_back(1);
    split_points_i.push_back(3);
    spb_i.split(split_points_i);

    //Bispace for j 
    sparse_bispace<1> spb_j(4);
    std::vector<size_t> split_points_j;
    split_points_j.push_back(1);
    split_points_j.push_back(3);
    spb_j.split(split_points_j);

    //Bispace for k 
    sparse_bispace<1> spb_k(5);
    std::vector<size_t> split_points_k;
    split_points_k.push_back(2);
    split_points_k.push_back(3);
    spb_k.split(split_points_k);

    size_t seq_0_arr_1[3] = {0,1,2};
    size_t seq_1_arr_1[3] = {1,0,1};
    size_t seq_2_arr_1[3] = {1,1,1};
    size_t seq_3_arr_1[3] = {2,0,0};
    size_t seq_4_arr_1[3] = {2,1,1};
    size_t seq_5_arr_1[3] = {2,1,2};

    std::vector< sequence<3,size_t> > ij_sig_blocks(6);
    for(size_t i = 0; i < 3; ++i) ij_sig_blocks[0][i] = seq_0_arr_1[i];
    for(size_t i = 0; i < 3; ++i) ij_sig_blocks[1][i] = seq_1_arr_1[i];
    for(size_t i = 0; i < 3; ++i) ij_sig_blocks[2][i] = seq_2_arr_1[i];
    for(size_t i = 0; i < 3; ++i) ij_sig_blocks[3][i] = seq_3_arr_1[i];
    for(size_t i = 0; i < 3; ++i) ij_sig_blocks[4][i] = seq_4_arr_1[i];
    for(size_t i = 0; i < 3; ++i) ij_sig_blocks[5][i] = seq_5_arr_1[i];

    sparse_bispace<3> spb_A  = spb_i % spb_j % spb_k << ij_sig_blocks;
    sparse_btensor<3> A(spb_A,A_arr,true);

    //Use identity matrix to simplify test
    sparse_bispace<2> spb_eye = spb_k|spb_k;
    double* eye_arr = new double[spb_eye.get_nnz()];
    memset(eye_arr,0,spb_eye.get_nnz()*sizeof(double));
    for(size_t i = 0; i < spb_k.get_dim(); ++i)
    {
        eye_arr[i*spb_k.get_dim()+i] = 1;
    }
    sparse_btensor<2> eye(spb_eye,eye_arr,false);
    delete [] eye_arr;

    direct_sparse_btensor<3> B(spb_A);
    letter i,j,k,l;
    B(i|j|l) = contract(k,A(i|j|k),eye(k|l));

    //We just contract with eye again to make things really simple
    sparse_btensor<3> C(spb_A);
    C(i|j|l) = contract(k,B(i|j|k),eye(k|l),20*sizeof(double),&j);
    sparse_btensor<3> C_correct(spb_A,A_arr,true);
    if(C != C_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "contract did not return correct value when batch index explicitly specified");
    }
}
#endif

} // namespace libtensor
