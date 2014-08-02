#include <libtensor/block_sparse/direct_sparse_btensor.h>
#include <libtensor/block_sparse/sparse_btensor.h>
#include <libtensor/block_sparse/direct_sparse_btensor_new.h>
#include <libtensor/block_sparse/sparse_btensor_new.h>
#include <libtensor/expr/operators/contract.h>
#include "direct_sparse_btensor_test.h"
#include "test_fixtures/contract2_test_f.h"
#include <math.h>

namespace libtensor {

void direct_sparse_btensor_test::perform() throw(libtest::test_exception) {
    /*test_contract2_direct_rhs();*/
    /*test_contract2_subtract2_nested();*/
    /*test_contract2_permute_nested();*/
    /*test_custom_batch_provider();*/
    /*test_force_batch_index();*/
}

//TODO: group with other test in test fixture
void direct_sparse_btensor_test::test_contract2_direct_rhs() throw(libtest::test_exception)
{
    static const char *test_name = "direct_sparse_btensor_test::test_contract2_direct_rhs()";
    contract2_test_f tf;

    /*** FIRST STEP - SET UP DIRECT TENSOR ***/
    sparse_btensor_new<3> A(tf.spb_A,tf.A_arr,true);
    sparse_btensor_new<3> B(tf.spb_B,tf.B_arr,true);
    sparse_bispace<2> spb_C = tf.spb_i | tf.spb_l;

    direct_sparse_btensor_new<2> C(spb_C);

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

    //Make batch memory just big enough to fit i = 1 batch of C 
    //This will force partitioning into i = 0 and i = 1
    /*E(m|i) = contract(l,D(m|l),C(i|l),96);*/
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
}

#if 0
void direct_sparse_btensor_test::test_contract2_subtract2_nested() throw(libtest::test_exception)
{
    static const char *test_name = "direct_sparse_btensor_test::test_contract2_subtract2_nested()";

    contract2_test_f tf;

    /*** FIRST STEP - SET UP DIRECT TENSOR ***/
    sparse_btensor<3> A(tf.spb_A,tf.A_arr,true);
    sparse_btensor<3> B(tf.spb_B,tf.B_arr,true);
    sparse_bispace<2> spb_C = tf.spb_i | tf.spb_l;

    direct_sparse_btensor<2> C(spb_C);

    letter i,j,k,l;
    C(i|l) = contract(j|k,A(i|j|k),B(j|k|l));

    /*** SECOND STEP - SUBTRACTION OF DIRECT TENSOR ***/
    double F_arr[18] = { //i = 0 l = 0
                         1,2,
                         
                         //i = 0 l = 1
                         3,4,5,

                         //i = 0 l = 2
                         6,

                         //i = 1 l = 0
                         7,8,
                         9,10,
                         
                         //i = 1 l = 1
                         11,12,13,
                         14,15,16,
                         
                         //i = 1 l = 2
                         17,
                         18
                        };

    sparse_btensor<2> F(spb_C,F_arr,true);
    direct_sparse_btensor<2> G(spb_C);

    G(i|l) = C(i|l) - F(i|l);


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

    sparse_btensor<2> D(spb_D,D_arr,true);

    sparse_bispace<2> spb_E = spb_m | tf.spb_i;
    sparse_btensor<2> E(spb_E);
    letter m;

    //Make batch memory just big enough to fit i = 1 batch of C 
    //This will force partitioning into i = 0 and i = 1
    E(m|i) = contract(l,D(m|l),G(i|l),96);

    double E_correct_arr[18] = { //m = 0 i = 0
                                 21926,47151,72376,

                                 //m = 0 i = 1
                                 122364,133052, 
                                 240525,260660,
                                 358686,388268,

                                 //m = 1 i = 0
                                 55172,
                                 62381,
                                 69590,

                                 //m = 1 i = 1
                                 302222,329427,
                                 338618,369008,
                                 375014,408589
                               };


    sparse_btensor<2> E_correct(spb_E,E_correct_arr,true);
    if(E != E_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "contract(...) did not produce correct result");
    }
}

void direct_sparse_btensor_test::test_contract2_permute_nested() throw(libtest::test_exception)
{
    static const char *test_name = "direct_sparse_btensor_test::test_contract2_permute_nested()";

    contract2_test_f tf;
    /*** FIRST STEP - SET UP DIRECT TENSOR ***/
    sparse_btensor<3> A(tf.spb_A,tf.A_arr,true);
    sparse_btensor<3> B(tf.spb_B,tf.B_arr,true);
    sparse_bispace<2> spb_C = tf.spb_i | tf.spb_l;

    direct_sparse_btensor<2> C(spb_C);

    letter i,j,k,l;
    C(i|l) = contract(j|k,A(i|j|k),B(j|k|l));

    direct_sparse_btensor<2> C_perm(spb_C.permute(permutation<2>().permute(0,1)));
    C_perm(l|i) = C(i|l);

    double C_perm_batch_0_correct_arr[15] = { //l = 0 i = 0
                                              1640,1703,

                                              //l = 0 i = 1
                                              7853,8525,
                                              8056,8748,
                                         
                                              //l = 1 i = 0
                                              2661,2748,2835,

                                              //l = 1 i = 1
                                              12337,13313,
                                              12629,13630,
                                              12921,13947};

    double C_perm_batch_1_correct_arr[3] = { //l = 2 i = 0
                                             535,

                                             //l = 2 i = 1
                                             4625,5091};

    double C_perm_batch_arr[15] = {0};

    std::map<idx_pair,idx_pair> batches;
    batches[idx_pair(0,0)] = idx_pair(0,2);
    C_perm.get_batch(C_perm_batch_arr,batches,120);

    for(size_t i = 0; i < sizeof(C_perm_batch_0_correct_arr)/sizeof(C_perm_batch_0_correct_arr[0]); ++i)
    {
        if(C_perm_batch_arr[i] != C_perm_batch_0_correct_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "contract(...), then permute(...) did not produce correct result for batch 0");
        }
    }
    
    batches[idx_pair(0,0)] = idx_pair(2,3);
    C_perm.get_batch(C_perm_batch_arr,batches,24);
    for(size_t i = 0; i < sizeof(C_perm_batch_1_correct_arr)/sizeof(C_perm_batch_1_correct_arr[0]); ++i)
    {
        if(C_perm_batch_arr[i] != C_perm_batch_1_correct_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "contract(...), then permute(...) did not produce correct result for batch 1");
        }
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
    class two_n_fibonnaci_batch_provider : public batch_provider<double>
    {
    private:
        static std::vector<block_loop> make_loops(const std::vector<sparse_bispace_any_order>& bispaces)
        {
            std::vector<block_loop> loops;
            block_loop bl_0(bispaces);
            bl_0.set_subspace_looped(0,0);
            loops.push_back(bl_0);
            block_loop bl_1(bispaces);
            bl_1.set_subspace_looped(0,1);
            loops.push_back(bl_1);
            return loops;
        }
        size_t fibonnaci(size_t n)
        { 
            if(n == 0) { return 0; }
            else if(n == 1) { return 1; }
            else { return fibonnaci(n - 1) + fibonnaci(n - 2); }
        }
    protected:
        virtual void run_impl(const std::vector<block_loop>& loops,
                              const idx_list& direct_tensors,
                              const std::vector<sparse_bispace_any_order>& truncated_bispaces,
                              const std::vector<double*>& ptrs,
                              const std::map<size_t,idx_pair>& loop_batches)
        {
            //We explicitly address everything since this is just a stub
            sparse_bispace<1> spb_i = this->m_bispaces[0][0];
            sparse_bispace<1> spb_j = this->m_bispaces[0][1];
            size_t batch_off = 0;
            for(size_t i_block_idx = loop_batches.at(0).first; i_block_idx < loop_batches.at(0).second; ++i_block_idx)
            {
                size_t i_block_off = spb_i.get_block_abs_index(i_block_idx);
                size_t i_block_sz = spb_i.get_block_size(i_block_idx);
                for(size_t j_block_idx = 0; j_block_idx < spb_j.get_n_blocks(); ++j_block_idx)
                {
                    size_t j_block_off = spb_j.get_block_abs_index(j_block_idx);
                    size_t j_block_sz = spb_j.get_block_size(j_block_idx);
                    for(size_t i_element_idx = 0; i_element_idx < i_block_sz; ++i_element_idx)
                    {
                        for(size_t j_element_idx = 0; j_element_idx < j_block_sz; ++j_element_idx)
                        {
                            size_t two_n = (size_t) pow(2,i_block_off+i_element_idx);
                            size_t fibonnaci_n = fibonnaci(j_block_off+j_element_idx);
                            ptrs[0][batch_off] = two_n*fibonnaci_n;
                            ++batch_off;
                        }
                    }
                }
            }
        }

    public:
        two_n_fibonnaci_batch_provider(const std::vector<sparse_bispace_any_order>& bispaces) : batch_provider(make_loops(bispaces),
                                                                                                               bispaces,
                                                                                                                  std::vector<size_t>(1,0),
                                                                                                                  std::vector<batch_provider<double>* >(),
                                                                                                                  std::vector<double*>(1,NULL),
                                                                                                                  0) {}
        virtual batch_provider<double>* clone() const { return new two_n_fibonnaci_batch_provider(*this); }
    };

    std::vector<sparse_bispace_any_order> bispaces(1,spb_i|spb_j);
    direct_sparse_btensor<2> A(spb_i|spb_j);
    A.set_batch_provider(two_n_fibonnaci_batch_provider(bispaces));

    double A_batch_0_correct_arr[12] = { //i = 0 j = 0
                                         0,1,1,
                                         0,2,2,

                                        //i = 0 j = 1
                                        2,3,5,
                                        4,6,10};

    double A_batch_1_correct_arr[18] = { //i = 1 j = 0
                                         0,4,4,
                                         0,8,8,
                                         0,16,16,

                                         //i = 1 j = 1
                                         8,12,20,
                                         16,24,40,
                                         32,48,80};
                                        

    double A_batch_arr[18];

    std::map<idx_pair,idx_pair> batches;
    batches[idx_pair(0,0)] = idx_pair(0,1);
    A.get_batch(A_batch_arr,batches);
    for(size_t i = 0; i < sizeof(A_batch_0_correct_arr)/sizeof(A_batch_0_correct_arr[0]); ++i)
    {
        if(A_batch_arr[i] != A_batch_0_correct_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "Custom batch provider returned incorrect value for batch 0");
        }
    }

    batches[idx_pair(0,0)] = idx_pair(1,2);
    A.get_batch(A_batch_arr,batches);
    for(size_t i = 0; i < sizeof(A_batch_1_correct_arr)/sizeof(A_batch_1_correct_arr[0]); ++i)
    {
        if(A_batch_arr[i] != A_batch_1_correct_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "Custom batch provider returned incorrect value for batch 0");
        }
    }
}

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
