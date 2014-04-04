#include <libtensor/block_sparse/blas_isomorphism.h>
#include "blas_isomorphism_test.h"

using namespace std;

namespace libtensor {

void blas_isomorphism_test::perform() throw(libtest::test_exception) {
    test_matmul_isomorphism_params_identity_NN();
    test_matmul_isomorphism_params_identity_NT();
    test_matmul_isomorphism_params_identity_TN();
    test_matmul_isomorphism_params_identity_TT();
    test_matmul_isomorphism_params_3d_2d_A_perm();
    test_matmul_isomorphism_params_permuted_ioc();
    /*test_matmul_isomorphism_params_3d_3d_A_perm_B_perm();*/
    /*test_matmul_isomorphism_params_3d_3d_A_perm_B_perm();*/
}

//Give it NN,NT,TN,TT matrix multiply cases, it should return identity permutation for
//both bispaces for all of them
void blas_isomorphism_test::test_matmul_isomorphism_params_identity_NN() throw(libtest::test_exception)
{
    static const char *test_name = "blas_isomorphism_test::test_matmul_isomorphism_params_identity_NN()";

    //bispaces
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);

    vector<sparse_bispace_any_order> bispaces(1,spb_i | spb_j);
    bispaces.push_back(spb_i|spb_k);
    bispaces.push_back(spb_k|spb_j);

    vector<block_loop> loops(3,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,0);
    //j loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(2,1);
    //k loop
    loops[2].set_subspace_looped(1,1);
    loops[2].set_subspace_looped(2,0);

    sparse_loop_list sll(loops);

    matmul_isomorphism_params<double> mip(sll);

    runtime_permutation perm_A = mip.get_A_perm();
    runtime_permutation perm_B = mip.get_B_perm();
    bool A_trans = mip.get_A_trans();
    bool B_trans = mip.get_B_trans();


    runtime_permutation ident(2);
    if(perm_A != ident || perm_B != ident || A_trans != false || B_trans != false)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "matmul_isomorphism_params returned incorrect value for NN case");
    }
}

void blas_isomorphism_test::test_matmul_isomorphism_params_identity_NT() throw(libtest::test_exception)
{
    static const char *test_name = "blas_isomorphism_test::test_matmul_isomorphism_params_identity_NT()";

    //bispaces
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);

    vector<sparse_bispace_any_order> bispaces(1,spb_i | spb_j);
    bispaces.push_back(spb_i|spb_k);
    bispaces.push_back(spb_j|spb_k);

    vector<block_loop> loops(3,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,0);
    //j loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(2,0);
    //k loop
    loops[2].set_subspace_looped(1,1);
    loops[2].set_subspace_looped(2,1);

    sparse_loop_list sll(loops);

    matmul_isomorphism_params<double> mip(sll);

    runtime_permutation perm_A = mip.get_A_perm();
    runtime_permutation perm_B = mip.get_B_perm();
    bool A_trans = mip.get_A_trans();
    bool B_trans = mip.get_B_trans();


    runtime_permutation ident(2);
    if(perm_A != ident || perm_B != ident || A_trans != false || B_trans != true)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "matmul_isomorphism_params returned incorrect value for NT case");
    }
}

void blas_isomorphism_test::test_matmul_isomorphism_params_identity_TN() throw(libtest::test_exception)
{
    static const char *test_name = "blas_isomorphism_test::test_matmul_isomorphism_params_identity_TN()";

    //bispaces
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);

    vector<sparse_bispace_any_order> bispaces(1,spb_i | spb_j);
    bispaces.push_back(spb_k|spb_i);
    bispaces.push_back(spb_k|spb_j);

    vector<block_loop> loops(3,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,1);
    //j loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(2,1);
    //k loop
    loops[2].set_subspace_looped(1,0);
    loops[2].set_subspace_looped(2,0);

    sparse_loop_list sll(loops);

    matmul_isomorphism_params<double> mip(sll);

    runtime_permutation perm_A = mip.get_A_perm();
    runtime_permutation perm_B = mip.get_B_perm();
    bool A_trans = mip.get_A_trans();
    bool B_trans = mip.get_B_trans();


    runtime_permutation ident(2);
    if(perm_A != ident || perm_B != ident || A_trans != true || B_trans != false)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "matmul_isomorphism_params returned incorrect value for TN case");
    }
}

void blas_isomorphism_test::test_matmul_isomorphism_params_identity_TT() throw(libtest::test_exception)
{
    static const char *test_name = "blas_isomorphism_test::test_matmul_isomorphism_params_identity_TT()";

    //bispaces
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);

    vector< sparse_bispace_any_order > bispaces(1,spb_i | spb_j);
    bispaces.push_back(spb_k|spb_i);
    bispaces.push_back(spb_j|spb_k);

    vector<block_loop> loops(3,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,1);
    //j loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(2,0);
    //k loop
    loops[2].set_subspace_looped(1,0);
    loops[2].set_subspace_looped(2,1);

    sparse_loop_list sll(loops);


    matmul_isomorphism_params<double> mip(sll);

    runtime_permutation perm_A = mip.get_A_perm();
    runtime_permutation perm_B = mip.get_B_perm();
    bool A_trans = mip.get_A_trans();
    bool B_trans = mip.get_B_trans();


    runtime_permutation ident(2);
    if(perm_A != ident || perm_B != ident || A_trans != true || B_trans != true)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "matmul_isomorphism_params returned incorrect value for TT case");
    }
}

//Requires A permutation 102 (not 021 bcs that would wreck vectorization)
//C_ijl = \sum_k A_ikj B_kl
//dimensions: i = 2,j = 3,k = 4,l = 5
void blas_isomorphism_test::test_matmul_isomorphism_params_3d_2d_A_perm() throw(libtest::test_exception)
{
    static const char *test_name = "blas_isomorphism_test::test_matmul_isomorphism_params_3d_2d_A_perm()";

    //bispaces
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);
    sparse_bispace<1> spb_l(5);

    vector< sparse_bispace_any_order > bispaces(1,spb_i|spb_j|spb_l);
    bispaces.push_back(spb_i|spb_k|spb_j);
    bispaces.push_back(spb_k|spb_l);

    vector<block_loop> loops(4,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,0);

    //j loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(1,2);

    //l loop
    loops[2].set_subspace_looped(0,2);
    loops[2].set_subspace_looped(2,1);

    //k loop
    loops[3].set_subspace_looped(1,1);
    loops[3].set_subspace_looped(2,0);

    sparse_loop_list sll(loops);


    matmul_isomorphism_params<double> mip(sll);

    runtime_permutation perm_A = mip.get_A_perm();
    runtime_permutation perm_B = mip.get_B_perm();
    bool A_trans = mip.get_A_trans();
    bool B_trans = mip.get_B_trans();


    runtime_permutation correct_A_perm(3);
    correct_A_perm.permute(0,1);
    runtime_permutation ident(2);
    if(perm_A !=  correct_A_perm || perm_B != ident || A_trans != true || B_trans != false)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "matmul_isomorphism_params returned incorrect value for 3d_2d_A_perm case");
    }
}

//Cil = Aijk Blkj
//A is smaller, so it should permute A to match B
void blas_isomorphism_test::test_matmul_isomorphism_params_permuted_ioc() throw(libtest::test_exception)
{
    static const char *test_name = "blas_isomorphism_test::test_matmul_isomorphism_params_permuted_ioc()";

    //bispaces
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);
    sparse_bispace<1> spb_l(5);

    vector< sparse_bispace_any_order > bispaces(1,spb_i|spb_l);
    bispaces.push_back(spb_i|spb_j|spb_k);
    bispaces.push_back(spb_l|spb_k|spb_j);

    vector<block_loop> loops(4,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,0);
    //l loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(2,0);
    //j loop
    loops[2].set_subspace_looped(1,1);
    loops[2].set_subspace_looped(2,2);
    //k loop
    loops[3].set_subspace_looped(1,2);
    loops[3].set_subspace_looped(2,1);

    sparse_loop_list sll(loops);

    matmul_isomorphism_params<double> mip(sll);

    runtime_permutation perm_A = mip.get_A_perm();
    runtime_permutation perm_B = mip.get_B_perm();
    bool A_trans = mip.get_A_trans();
    bool B_trans = mip.get_B_trans();

    runtime_permutation correct_A_perm(3);
    correct_A_perm.permute(1,2);
    runtime_permutation ident(3);
    if(perm_A != correct_A_perm || perm_B != ident || A_trans != false || B_trans != true)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "matmul_isomorphism_params returned incorrect value for permuted_ioc case");
    }
}

#if 0
//Requires A permutation 021, B permutation 201
//Could also permute C additionally, but shouldn't, and therefore should reflect that
//C_ijml = \sum_k A_ikj B_lkm
//dimensions: i = 2,j = 3,k = 4,l = 5,m = 6
void blas_isomorphism_test::test_matmul_isomorphism_params_3d_3d_A_perm_B_perm() throw(libtest::test_exception)
{
    static const char *test_name = "blas_isomorphism_test::test_matmul_isomorphism_params_3d_3d_A_perm_B_perm()";

    //bispaces
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);
    sparse_bispace<1> spb_l(5);
    sparse_bispace<1> spb_m(6);

    vector< sparse_bispace_any_order > bispaces(1,spb_i|spb_j|spb_m|spb_l);
    bispaces.push_back(spb_i|spb_k|spb_j);
    bispaces.push_back(spb_l|spb_k|spb_m);

    vector<block_loop> loops(5,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,0);

    //j loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(1,2);

    //l loop
    loops[2].set_subspace_looped(0,3);
    loops[2].set_subspace_looped(2,0);

    //m loop
    loops[3].set_subspace_looped(0,2);
    loops[3].set_subspace_looped(2,2);

    //k loop
    loops[4].set_subspace_looped(1,1);
    loops[4].set_subspace_looped(2,1);


    sparse_loop_list sll(loops);


    matmul_isomorphism_params<double> mip(sll);

    runtime_permutation perm_A = mip.get_A_perm();
    runtime_permutation perm_B = mip.get_B_perm();
    matmul_isomorphism_params<double>::dgemm_fp_t dgemm_fp = mip.get_dgemm_fp();


    runtime_permutation correct_A_perm(3);
    correct_A_perm.permute(1,2);
    runtime_permutation correct_B_perm(2);
    correct_B_perm.permute(0,2);
    correct_B_perm.permute(1,2);
    if(perm_A !=  correct_A_perm || perm_B != correct_B_perm || dgemm_fp != &linalg::mul2_ij_ip_jp_x)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "matmul_isomorphism_params returned incorrect value for 3d_2d_A_perm case");
    }
}
#endif

} // namespace libtensor
