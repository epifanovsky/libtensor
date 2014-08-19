#include <libtensor/block_sparse/direct_sparse_btensor.h>
#include <libtensor/block_sparse/sparse_btensor.h>
#include <libtensor/expr/operators/contract.h>
#include "direct_sparse_btensor_test.h"
#include "test_fixtures/contract2_test_f.h"
#include "test_fixtures/contract2_subtract2_nested_test_f.h"
#include <math.h>
#include <fstream>
#include <iomanip>

using namespace std;

namespace libtensor {

void direct_sparse_btensor_test::perform() throw(libtest::test_exception) {
    test_contract2_direct_rhs();
    test_contract2_subtract2_nested();
    test_custom_batch_provider();
    test_assignment_chain();
    test_force_batch_index();
    test_pari_k();
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
    sparse_btensor<3> A(tf.spb_A,tf.A_arr,true);
    sparse_btensor<3> B(tf.spb_B,tf.B_arr,true);
    direct_sparse_btensor<2> C(tf.spb_C);

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

    sparse_btensor<2> D(spb_D,D_arr,true);

    sparse_bispace<2> spb_E = spb_m | tf.spb_i;
    sparse_btensor<2> E(spb_E);
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


    sparse_btensor<2> E_correct(spb_E,E_correct_arr,true);
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
    direct_sparse_btensor<2> A(spb_i|spb_j);
    two_n_fibonnaci_batch_provider bp(spb_i,spb_j);
    A.set_batch_provider(bp);
    sparse_btensor<2> B(spb_i|spb_j);
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
                                        

    sparse_btensor<2> B_correct(spb_i|spb_j,B_correct_arr,true);
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

    //TODO: Test assign away batch provider with normal tensor expr!!!
}

//Make sure that interconverting between direct and in-core representations of the same tensor is seamless
void direct_sparse_btensor_test::test_assignment_chain() throw(libtest::test_exception)
{
    static const char *test_name = "direct_sparse_btensor_test::test_assignment_chain()";

    contract2_subtract2_nested_test_f tf;
    direct_sparse_btensor<3> A_direct(tf.A.get_bispace());
    sparse_btensor<3> B(tf.A.get_bispace());
    letter i,j,k;
    A_direct(i|j|k) = tf.A(i|j|k);
    B(i|j|k) = A_direct(i|j|k);
    if(B != tf.A)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "assignment chain resulted in incorrect result");
    }
}

//Due to constraints of external code that may be providing us with some batches
//we may need to force the choice of a particular index to batch over.
void direct_sparse_btensor_test::test_force_batch_index() throw(libtest::test_exception)
{
    static const char *test_name = "direct_sparse_btensor_test::test_force_batch_index()";


    class A_mock : public batch_provider_i<double>
    {
    public:
        sparse_bispace<3> m_A_bispace;
        //Block major
        double A_arr[26];

        //Force batching over J
        virtual idx_list get_batchable_subspaces() const { return idx_list(1,1); }
        virtual void get_batch(double* output_ptr,const bispace_batch_map& bbm = bispace_batch_map()) 
        {
            bispace_batch_map correct_bbm;
            if(bbm.size() != 1 || (bbm.find(idx_pair(0,1)) == bbm.end()))
            {
                throw bad_parameter(g_ns, "A_mock","get_batch",__FILE__, __LINE__, "Invalid batching!");
            }
            idx_pair batch = bbm.find(idx_pair(0,1))->second;
            sparse_block_tree_any_order A_tree = m_A_bispace.get_sparse_group_tree(0);
            A_tree = A_tree.truncate_subspace(1,batch);
            for(sparse_block_tree_any_order::iterator it = A_tree.begin(); it != A_tree.end(); ++it)
            {
                for(size_t i = 0; i < (*it)[0].second; ++i)
                {
                    *(output_ptr++) = A_arr[(*it)[0].first+i];
                }
            }
        }

        A_mock(const sparse_bispace<3>& spb_A) : m_A_bispace(spb_A)
        {
            for(size_t i = 0; i < sizeof(A_arr)/sizeof(A_arr[0]); ++i)
            {
                A_arr[i] = i+1;
            }
        }
    };

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
    A_mock am(spb_A);
    direct_sparse_btensor<3> A(spb_A);
    A.set_batch_provider(am);

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
    memory_reserve mr_0((26+2*20)*sizeof(double));
    sparse_btensor<3> C(spb_A);
    C.set_memory_reserve(mr_0);
    C(i|j|l) = contract(k,B(i|j|k),eye(k|l));
    sparse_btensor<3> C_correct(spb_A,am.A_arr,true);
    if(C != C_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "contract did not return correct value when batch index explicitly specified");
    }
}

//This test implements the electronic structure PARI-K algorithm
void direct_sparse_btensor_test::test_pari_k() throw(libtest::test_exception)
{
    static const char *test_name = "direct_sparse_btensor_test::test_pari_k()";

    //Enough memory for all full tensors + first block of Q index
    memory_reserve mr_0((40432+67883+20216)*8);
    memory_reserve mr_1((40432+67883+20216)*8-1);
    memory_reserve mr_2(1e10);

    size_t N;
    size_t X;
    size_t o;
    vector<size_t> split_points_N;
    vector<size_t> split_points_X;
    vector< sequence<3,size_t> > sig_blocks_NNX;

    ifstream C_ifs;
    C_ifs.open("../tests/block_sparse/test_fixtures/pari_k_C.txt");
    string line;
    getline(C_ifs,line);
    istringstream(line) >> N;
    getline(C_ifs,line);
    istringstream(line) >> X;
    getline(C_ifs,line);
    istringstream(line) >> o;

    //Read N splitting information
    //Skip section header 
    getline(C_ifs,line);
    getline(C_ifs,line);
    while(line.length() > 0)
    {
        size_t entry;
        istringstream(line) >> entry;
        split_points_N.push_back(entry);
        getline(C_ifs,line);
    }

    //Read the X splitting information  
    getline(C_ifs,line);
    while(line.length() > 0)
    {
        size_t entry;
        istringstream(line) >> entry;
        split_points_X.push_back(entry);
        getline(C_ifs,line);
    }

    //Get the shell-shell-aux atom sparsity information
    getline(C_ifs,line);
    while(line.length() > 0)
    {
        sequence<3,size_t> entry;
        istringstream(line) >> entry[0] >> entry[1] >> entry[2];
        sig_blocks_NNX.push_back(entry);
        getline(C_ifs,line);
    }

    sparse_bispace<1> spb_N(N);
    spb_N.split(split_points_N);
    sparse_bispace<1> spb_X(X);
    spb_X.split(split_points_X);
    sparse_bispace<1> spb_o(o);

    sparse_bispace<3> spb_C = spb_N % spb_N % spb_X << sig_blocks_NNX;

    //Read C entries
    double* C_arr = new double[spb_C.get_nnz()];
    getline(C_ifs,line);
    size_t C_idx = 0;
    while(line.length() > 0)
    {
        istringstream(line) >> C_arr[C_idx];
        ++C_idx;
        getline(C_ifs,line);
    }
    sparse_btensor<3> C(spb_C,C_arr,true);
    delete [] C_arr;

    //Read I entries
    sparse_bispace<3> spb_I = spb_C.contract(2) | spb_X;
    double* I_arr = new double[spb_I.get_nnz()];
    ifstream I_ifs;
    I_ifs.open("../tests/block_sparse/test_fixtures/pari_k_I.txt");
    getline(I_ifs,line);
    size_t I_idx = 0;
    while(line.length() > 0)
    {
        istringstream(line) >> I_arr[I_idx];
        ++I_idx;
        getline(I_ifs,line);
    }

    //We fake 'I' being direct so that we can - only one of the 'I' copies is accounted for in the memory total above
    sparse_btensor<3> I_fake_0(spb_I,I_arr,true);
    sparse_btensor<3> I_fake_1(spb_I.permute(permutation<3>().permute(0,1)));
    delete [] I_arr;

    //Read V_scaled entries
    sparse_bispace<2> spb_V_scaled = spb_X|spb_X;  
    double* V_scaled_arr = new double[spb_V_scaled.get_nnz()];
    ifstream V_scaled_ifs;
    V_scaled_ifs.open("../tests/block_sparse/test_fixtures/pari_k_V_scaled.txt");
    getline(V_scaled_ifs,line);
    size_t V_scaled_idx = 0;
    while(line.length() > 0)
    {
        istringstream(line) >> V_scaled_arr[V_scaled_idx];
        ++V_scaled_idx;
        getline(V_scaled_ifs,line);
    }
    sparse_btensor<2> V_scaled(spb_V_scaled,V_scaled_arr,true);
    delete [] V_scaled_arr;

    //Read C_mo entries
    ifstream C_mo_ifs;
    C_mo_ifs.open("../tests/block_sparse/test_fixtures/pari_k_C_mo.txt");
    sparse_bispace<2> spb_C_mo = spb_N|spb_o;
    double* C_mo_arr = new double[spb_C_mo.get_nnz()];
    getline(C_mo_ifs,line);
    size_t C_mo_idx = 0;
    while(line.length() > 0)
    {
        istringstream(line) >> C_mo_arr[C_mo_idx];
        ++C_mo_idx;
        getline(C_mo_ifs,line);
    }
    sparse_btensor<2> C_mo(spb_C_mo,C_mo_arr,true);
    delete [] C_mo_arr;

    sparse_btensor<3> C_perm(spb_C.permute(permutation<3>().permute(1,2)));
    direct_sparse_btensor<3> D(C_perm.get_bispace().contract(2)|spb_o);
    direct_sparse_btensor<3> E(spb_C.contract(2)|spb_X);
    direct_sparse_btensor<3> I(E.get_bispace());
    direct_sparse_btensor<3> G(E.get_bispace());
    direct_sparse_btensor<3> H(spb_N|spb_X|spb_o);
    sparse_btensor<2> M(spb_N|spb_N);

    C.set_memory_reserve(mr_0);
    C_perm.set_memory_reserve(mr_0);
    I_fake_0.set_memory_reserve(mr_0);
    M.set_memory_reserve(mr_0);


    letter mu,nu,lambda,sigma,Q,R,i;
    C_perm(mu|Q|lambda) = C(mu|lambda|Q);
    I_fake_1(sigma|nu|Q) = I_fake_0(nu|sigma|Q);
    I(nu|sigma|Q) = I_fake_1(sigma|nu|Q);
    D(mu|Q|i) = contract(lambda,C_perm(mu|Q|lambda),C_mo(lambda|i));
    E(nu|sigma|Q) = contract(R,C(nu|sigma|R),V_scaled(Q|R));
    G(nu|sigma|Q) = I(nu|sigma|Q) - E(nu|sigma|Q);
    H(nu|Q|i) = contract(sigma,G(nu|sigma|Q),C_mo(sigma|i));
    M(mu|nu) = contract(Q|i,D(mu|Q|i),H(nu|Q|i));

    sparse_bispace<2> spb_M = spb_N|spb_N;
    double* M_correct_arr = new double[spb_M.get_nnz()];
    ifstream M_correct_ifs;
    M_correct_ifs.open("../tests/block_sparse/test_fixtures/pari_k_M.txt");
    getline(M_correct_ifs,line);
    size_t M_correct_idx = 0;

    while(line.length() > 0)
    {
        istringstream(line) >> M_correct_arr[M_correct_idx];
        ++M_correct_idx;
        getline(M_correct_ifs,line);
    }
    sparse_btensor<2> M_correct(spb_M,M_correct_arr,true);
    delete [] M_correct_arr;

    if(M != M_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "pari_k result incorrect");
    }

    //Check with no batching
    C.set_memory_reserve(mr_2);
    C_perm.set_memory_reserve(mr_2);
    I_fake_0.set_memory_reserve(mr_2);
    M.set_memory_reserve(mr_2);
    memset((double*)M.get_data_ptr(),0,M.get_bispace().get_nnz()*sizeof(double));
    M(mu|nu) = contract(Q|i,D(mu|Q|i),H(nu|Q|i));
    if(M != M_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "pari_k result incorrect when full unbatched");
    }

    C.set_memory_reserve(mr_1);
    C_perm.set_memory_reserve(mr_1);
    I_fake_0.set_memory_reserve(mr_1);
    M.set_memory_reserve(mr_1);
    bool threw_exception = false;
    try
    {
        M(mu|nu) = contract(Q|i,D(mu|Q|i),H(nu|Q|i));
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

} // namespace libtensor
