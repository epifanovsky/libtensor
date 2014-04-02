#include <libtensor/block_sparse/blas_isomorphism.h>
#include "blas_isomorphism_test.h"

using namespace std;

namespace libtensor {

void blas_isomorphism_test::perform() throw(libtest::test_exception) {
    test_matmul_isomorphism_params_identity_NN();
    test_matmul_isomorphism_params_identity_NT();
    test_matmul_isomorphism_params_identity_TN();
    test_matmul_isomorphism_params_identity_TT();
}

//Give it NN,NT,TN,TT matrix multiply cases, it should return identity permutation for
//both bispaces for all of them
void blas_isomorphism_test::test_matmul_isomorphism_params_identity_NN() throw(libtest::test_exception)
{
    static const char *test_name = "blas_isomorphism_test::test_matmul_perm_identity_NN()";

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

    runtime_permutation perm_A = mip.get_perm_A();
    runtime_permutation perm_B = mip.get_perm_B();
    matmul_isomorphism_params<double>::dgemm_fp_t dgemm_fp = mip.get_dgemm_fp();


    runtime_permutation ident(2);
    if(perm_A != ident || perm_B != ident || dgemm_fp != &linalg::mul2_ij_ip_pj_x)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "matmul_isomorphism_params returned incorrect value for NN case");
    }
}

void blas_isomorphism_test::test_matmul_isomorphism_params_identity_NT() throw(libtest::test_exception)
{
    static const char *test_name = "blas_isomorphism_test::test_matmul_perm_identity_NT()";

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

    runtime_permutation perm_A = mip.get_perm_A();
    runtime_permutation perm_B = mip.get_perm_B();
    matmul_isomorphism_params<double>::dgemm_fp_t dgemm_fp = mip.get_dgemm_fp();


    runtime_permutation ident(2);
    if(perm_A != ident || perm_B != ident || dgemm_fp != &linalg::mul2_ij_ip_jp_x)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "matmul_isomorphism_params returned incorrect value for NT case");
    }
}

void blas_isomorphism_test::test_matmul_isomorphism_params_identity_TN() throw(libtest::test_exception)
{
    static const char *test_name = "blas_isomorphism_test::test_matmul_perm_identity_TN()";

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

    runtime_permutation perm_A = mip.get_perm_A();
    runtime_permutation perm_B = mip.get_perm_B();
    matmul_isomorphism_params<double>::dgemm_fp_t dgemm_fp = mip.get_dgemm_fp();


    runtime_permutation ident(2);
    if(perm_A != ident || perm_B != ident || dgemm_fp != &linalg::mul2_ij_pi_pj_x)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "matmul_isomorphism_params returned incorrect value for TN case");
    }
}

void blas_isomorphism_test::test_matmul_isomorphism_params_identity_TT() throw(libtest::test_exception)
{
    static const char *test_name = "blas_isomorphism_test::test_matmul_perm_identity_TT()";

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

    runtime_permutation perm_A = mip.get_perm_A();
    runtime_permutation perm_B = mip.get_perm_B();
    matmul_isomorphism_params<double>::dgemm_fp_t dgemm_fp = mip.get_dgemm_fp();


    runtime_permutation ident(2);
    if(perm_A != ident || perm_B != ident || dgemm_fp != &linalg::mul2_ij_pi_jp_x)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "matmul_isomorphism_params returned incorrect value for TT case");
    }
}

} // namespace libtensor
