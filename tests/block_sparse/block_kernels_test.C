#include <libtensor/block_sparse/sparse_loop_list.h>
#include <libtensor/block_sparse/block_print_kernel.h>
#include <libtensor/block_sparse/block_permute_kernel.h>
#include <libtensor/block_sparse/block_contract2_kernel.h>
#include <libtensor/block_sparse/block_subtract2_kernel.h>
#include <libtensor/expr/iface/letter.h>
#include <sstream>
#include "block_kernels_test.h" 

using namespace std;

namespace libtensor {

void block_kernels_test::perform() throw(libtest::test_exception) {

    test_block_print_kernel_2d();
    test_block_print_kernel_3d();

    test_block_permute_kernel_2d();
    test_block_permute_kernel_3d_120();
    test_block_permute_kernel_3d_021();

    test_block_contract2_kernel_2d_not_enough_loops();
    test_block_contract2_kernel_2d_not_enough_bispaces();
    test_block_contract2_kernel_2d_C_missing_idx();
    test_block_contract2_kernel_2d_C_extra_idx();
    test_block_contract2_kernel_2d_no_contracted_inds();
    test_block_contract2_kernel_2d_strided_output();
    test_block_contract2_kernel_non_matmul_A_ikj();
    test_block_contract2_kernel_non_matmul_A_jik();
    test_block_contract2_kernel_non_matmul_A_trans_kjli();
    test_block_contract2_kernel_non_matmul_A_trans_klji();
    test_block_contract2_kernel_non_matmul_B_jlk();
    test_block_contract2_kernel_non_matmul_B_jkml();
    test_block_contract2_kernel_non_matmul_B_trans_kjl();
    test_block_contract2_kernel_non_matmul_B_trans_lkj();

    test_block_contract2_kernel_2d_not_enough_dims_and_ptrs();
    test_block_contract2_kernel_2d_invalid_dims();
    test_block_contract2_kernel_2d_incompatible_dims();
    test_block_contract2_kernel_2d_ik_kj();
    test_block_contract2_kernel_2d_ik_jk();
    test_block_contract2_kernel_2d_ki_kj();
    test_block_contract2_kernel_2d_ki_jk();
    test_block_contract2_kernel_2d_ki_kj_permuted_loops();
    test_block_contract2_kernel_3d_2d();
    test_block_contract2_kernel_3d_3d_multi_index();
    test_block_contract2_kernel_matrix_vector_mult();

    test_block_subtract2_kernel_not_enough_dims_and_ptrs();
    test_block_subtract2_kernel_invalid_dims();
    test_block_subtract2_kernel_2d();
}


void block_kernels_test::test_block_print_kernel_2d() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_print_kernel_2d()";

    double test_block_arr[4] = {1,2,
                                3,4};
    vector<double*> ptrs(1,test_block_arr);

    dim_list dims;
    dims.push_back(2); 
    dims.push_back(2); 
    vector<dim_list> dim_lists(1,dims);

    block_print_kernel<double> bpk;
    bpk(ptrs,dim_lists);
    string correct_str("---\n 1 2\n 3 4\n");

    if(bpk.str()  != correct_str)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_printer::str(...) returned incorrect value");
    }
}


void block_kernels_test::test_block_print_kernel_3d() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_print_kernel_3d()";

    double test_block_arr[8] = {1,2,
                                3,4,
 
                                5,6,
                                7,8};
    vector<double*> ptrs(1,test_block_arr);

    dim_list dims;
    dims.push_back(2); 
    dims.push_back(2); 
    dims.push_back(2); 
    vector<dim_list> dim_lists(1,dims);

    block_print_kernel<double> bpk;
    bpk(ptrs,dim_lists);
    string correct_str("---\n 1 2\n 3 4\n\n 5 6\n 7 8\n");
    if(bpk.str()  != correct_str)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_printer::str(...) returned incorrect value");
    }
}

void block_kernels_test::test_block_permute_kernel_2d() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d()";

    double test_input_block[4] = {1,2,
                                  3,4};

    double correct_output_block[4] = {1,3,
                                      2,4};
    double test_output_block[4];

    vector<double*> ptrs;
    ptrs.push_back(test_output_block);
    ptrs.push_back(test_input_block);

    dim_list dims;
    dims.push_back(2);
    dims.push_back(2);
    vector<dim_list> dim_lists(2,dims);

    runtime_permutation perm(2);
    perm.permute(0,1);
    block_permute_kernel<double> b_perm_k(perm);
    b_perm_k(ptrs,dim_lists);

    for(int i = 0; i < 4; ++i)
    {
        if(test_output_block[i] != correct_output_block[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_permute_kernel::operator(...) did not produce correct result");
        }
    }
}

//A more complicated 3D permutation, with indices permuted in cyclic order
void block_kernels_test::test_block_permute_kernel_3d_120() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_3d_201()";

    //ordering is kij slow->fast to start
    //Dimensions are: 
    //k = 4
    //i = 2
    //j = 3
    

    //kij
    double test_input_block[24] = { //k = 0 
                                       1,2,3, //i = 0
                                       4,5,6, //i = 1

                                       //k = 1
                                       7,8,9, //i = 0
                                       10,11,12, //i = 1

                                       //k = 2 
                                       13,14,15, //i = 0
                                       16,17,18,//i = 1

                                       //k = 3
                                       19,20,21, //i = 0
                                       22,23,24 //i = 1
                                     };

    //ijk
    double correct_output_block[24] = { //i = 0; 
                                 1,7,13,19, //j = 0
                                 2,8,14,20, //j = 1
                                 3,9,15,21, //j=2

                                 //i = 1
                                 4,10,16,22, //j = 0
                                 5,11,17,23, //j = 1
                                 6,12,18,24}; // j = 2

    double test_output_block[24];

    vector<double*> ptrs;
    ptrs.push_back(test_output_block);
    ptrs.push_back(test_input_block);


    runtime_permutation perm(3);
    perm.permute(0,2);
    perm.permute(0,1);

    dim_list dims;
    dims.push_back(4);
    dims.push_back(2);
    dims.push_back(3);
    vector<dim_list> dim_lists(2);
    dim_lists[1] = dims;
    perm.apply(dims);
    dim_lists[0] = dims;

    block_permute_kernel<double> b_perm_k(perm);
    b_perm_k(ptrs,dim_lists);
    for(int i = 0; i < 24; ++i)
    {
        if(test_output_block[i] != correct_output_block[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_permute_kernel::operator(...) did not produce correct result");
        }
    }
}

void block_kernels_test::test_block_permute_kernel_3d_021() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_3d_021()";

    //ordering is kij slow->fast to start
    //Dimensions are: 
    //k = 4
    //i = 2
    //j = 3

    //kij
    double test_input_block[24] = { //k = 0 
                                       1,2,3, //i = 0
                                       4,5,6, //i = 1

                                       //k = 1
                                       7,8,9, //i = 0
                                       10,11,12, //i = 1

                                       //k = 2 
                                       13,14,15, //i = 0
                                       16,17,18,//i = 1

                                       //k = 3
                                       19,20,21, //i = 0
                                       22,23,24 //i = 1
                                   };

    //kji
    double correct_output_block[24] = { //k = 0 
                                       1,4, //j=0
                                       2,5, //j=1
                                       3,6, //j=2

                                       //k = 1
                                       7,10, //j=0
                                       8,11, //j=1
                                       9,12, //j=2

                                       //k = 2
                                       13,16, //j=0
                                       14,17, //j=1
                                       15,18, //j=2

                                       //k = 3
                                       19,22, //j=0
                                       20,23, //j=1
                                       21,24  //j=2
                                      };

    double test_output_block[24];

    vector<double*> ptrs;
    ptrs.push_back(test_output_block);
    ptrs.push_back(test_input_block);

    dim_list input_dims;
    input_dims.push_back(4);
    input_dims.push_back(2);
    input_dims.push_back(3);

    dim_list output_dims;
    output_dims.push_back(4);
    output_dims.push_back(3);
    output_dims.push_back(2);

    vector<dim_list> dim_lists;
    dim_lists.push_back(output_dims);
    dim_lists.push_back(input_dims);

    runtime_permutation perm(3);
    perm.permute(1,2);
    block_permute_kernel<double> b_perm_k(perm);
    b_perm_k(ptrs,dim_lists);

    for(int i = 0; i < 24; ++i)
    {
        if(test_output_block[i] != correct_output_block[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_permute_kernel::operator(...) did not produce correct result");
        }
    }
}

//Should throw an exception because there is only one loop, minimum is two loops for matrix-vector multiply
void block_kernels_test::test_block_contract2_kernel_2d_not_enough_loops() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_contract2_kernel_2d_not_enough_loops()";

    //C_i = \sum_k A_ij B_j
    //dimensions: i = 2,j = 3
    //Just need dummy bispaces for this test
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);

    vector< sparse_bispace_any_order > bispaces(1,spb_i);
    bispaces.push_back(spb_i|spb_j);
    bispaces.push_back(spb_j);

    //i loop
    vector<block_loop> loops(1,block_loop(bispaces));
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,0);

    //j loop - left out!

    sparse_loop_list sll(loops);

    bool threw_exception = false;
    try
    {
        block_contract2_kernel<double> bc2k(sll);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel(...) did not throw exception when less than three loops specified");
    }
}

//Should throw an exception exception when there are too few bispaces
void block_kernels_test::test_block_contract2_kernel_2d_not_enough_bispaces() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_not_enough_bispaces()";

    //C_ij = \sum_k A_ik B_kj
    //dimensions: i = 2,j = 3,k = 4
    //Just need dummy bispaces for this test
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);

    vector< sparse_bispace_any_order > bispaces(1,spb_i | spb_j);
    bispaces.push_back(spb_i|spb_k);
    //DELIBERATELY LEAVE OUT B bispaces

    vector<block_loop> loops(3,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,0);
    //j loop
    loops[1].set_subspace_looped(0,1);
    //k loop
    loops[2].set_subspace_looped(1,1);

    sparse_loop_list sll(loops);

    bool threw_exception = false;
    try
    {
    	block_contract2_kernel<double> bc2k(sll);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when not enough bispaces specified");
    }
}

//If a loop ignores C, the index over which it loops must be contracted
//meaning that it must touch both A and B. If it does not touch one of them, the
//index should have appeared in C
void block_kernels_test::test_block_contract2_kernel_2d_C_missing_idx() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_contract2_kernel_2d_C_missing_idx()";

    //C_ij = \sum_k A_ik B_kj
    //dimensions: i = 2,j = 3,k = 4
    //Just need dummy bispaces for this test
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);

    vector< sparse_bispace_any_order > bispaces(1,spb_i | spb_j);
    bispaces.push_back(spb_i|spb_k);
    bispaces.push_back(spb_k|spb_j);

    vector<block_loop> loops(3,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,0);

    //j loop
    //DELIBERATELY DONT TOUCH j INDEX OF C - SHOULD CAUSE AN ERROR BECAUSE J IS NOT CONTRACTED
    loops[1].set_subspace_looped(2,1);

    //k loop
    loops[2].set_subspace_looped(1,1);
    loops[2].set_subspace_looped(2,0);

    sparse_loop_list sll(loops);

    bool threw_exception = false;
    try
    {
    	block_contract2_kernel<double> bc2k(sll);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when an uncontracted index missing from C");
    }
}

void block_kernels_test::test_block_contract2_kernel_2d_C_extra_idx() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_contract2_kernel_2d_C_extra_idx()";

    //C_ij = \sum_k A_ik B_kj
    //dimensions: i = 2,j = 3,k = 4
    //Just need dummy bispaces for this test
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);

    vector< sparse_bispace_any_order > bispaces(1,spb_i | spb_j);
    bispaces.push_back(spb_i|spb_k);
    bispaces.push_back(spb_k|spb_j);

    vector<block_loop> loops(3,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    //DELIBERATELY DON'T TOUCH i occurence in A - should cause an exception

    //j loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(2,1);

    //k loop
    loops[2].set_subspace_looped(1,1);
    loops[2].set_subspace_looped(2,0);

    sparse_loop_list sll(loops);

    bool threw_exception = false;
    try
    {
    	block_contract2_kernel<double> bc2k(sll);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when an index in C but not A or B");
    }
}

//C_ijk = A_jik + B_jik
//dimensions: i = 2,j = 3,k = 4
//No contracted indices, so should throw an exception
void block_kernels_test::test_block_contract2_kernel_2d_no_contracted_inds() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_contract2_kernel_2d_no_contracted_inds()";

    //Just need dummy bispaces for this test
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);

    vector<sparse_bispace_any_order> bispaces(1,spb_i | spb_j | spb_k);
    bispaces.push_back(spb_j| spb_i | spb_k);
    bispaces.push_back(spb_j| spb_i | spb_k);

    vector<block_loop> loops(3,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,1);
    loops[0].set_subspace_looped(2,1);

    //j loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(1,0);
    loops[1].set_subspace_looped(2,0);

    //k loop
    loops[2].set_subspace_looped(0,2);
    loops[2].set_subspace_looped(1,2);
    loops[2].set_subspace_looped(2,2);

    sparse_loop_list sll(loops);

    bool threw_exception = false;
    try
    {
    	block_contract2_kernel<double> bc2k(sll);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when no contracted indices present");
    }
}

//We don't allow output to be strided in the inner kernel - just dumb...and unsupported by typical blas implementations
void block_kernels_test::test_block_contract2_kernel_2d_strided_output() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_strided_output()";

    //Output strided test
    //C_ji = \sum_k A_ik B_kj
    //dimensions: i = 2,j = 3,k = 4
    //Just need dummy bispaces for this test
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);

    vector< sparse_bispace_any_order > bispaces(1,spb_j | spb_i);
    bispaces.push_back(spb_i|spb_k);
    bispaces.push_back(spb_k|spb_j);

    vector<block_loop> loops(3,block_loop(bispaces));
    //Deliberately swap i and j order
    //j loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(2,1);
    //i loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(1,0);
    //k loop
    loops[2].set_subspace_looped(1,1);
    loops[2].set_subspace_looped(2,0);

    sparse_loop_list sll(loops);

    bool threw_exception = false;
    try
    {
    	block_contract2_kernel<double> bc2k(sll);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when output is strided");
    }
}

//C_ijl = \sum_k A_ikj B_kl
//Invalid because contracted and uncontracted indices mixed
//dimensions: i = 2,j = 3,k = 4,l = 5
void block_kernels_test::test_block_contract2_kernel_non_matmul_A_ikj() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_contract2_kernel_non_matmul_A_ikj()";

    //Just need dummy bispaces for this test
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

    bool threw_exception = false;
    try
    {
    	block_contract2_kernel<double> bc2k(sll);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when matmul-incompatible order specified");
    }

}

//C_ijl = \sum_k A_jik B_kl
//Invalid because last index of A appearing in C does not appear in A immediately before the contracted indices
//dimensions: i = 2,j = 3,k = 4,l = 5
void block_kernels_test::test_block_contract2_kernel_non_matmul_A_jik() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_contract2_kernel_non_matmul_A_jik()";

    //Just need dummy bispaces for this test
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);
    sparse_bispace<1> spb_l(5);

    vector< sparse_bispace_any_order > bispaces(1,spb_i|spb_j|spb_l);
    bispaces.push_back(spb_j|spb_i|spb_k);
    bispaces.push_back(spb_k|spb_l);

    vector<block_loop> loops(4,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,1);

    //j loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(1,0);

    //l loop
    loops[2].set_subspace_looped(0,2);
    loops[2].set_subspace_looped(2,1);

    //k loop
    loops[3].set_subspace_looped(1,2);
    loops[3].set_subspace_looped(2,0);

    sparse_loop_list sll(loops);

    bool threw_exception = false;
    try
    {
    	block_contract2_kernel<double> bc2k(sll);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when matmul-incompatible order specified");
    }
}

//C_ijm = \sum_kl A_kjli B_klm
//Invalid because contracted and uncontracted indices mixed
//dimensions: i = 2,j = 3,k = 4,l = 5,m = 6
void block_kernels_test::test_block_contract2_kernel_non_matmul_A_trans_kjli() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_contract2_kernel_non_matmul_A_trans_kjli()";

    //Just need dummy bispaces for this test
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);
    sparse_bispace<1> spb_l(5);
    sparse_bispace<1> spb_m(6);

    vector< sparse_bispace_any_order > bispaces(1,spb_i|spb_j|spb_m);
    bispaces.push_back(spb_k|spb_j|spb_l|spb_i);
    bispaces.push_back(spb_k|spb_l|spb_m);

    vector<block_loop> loops(5,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,3);

    //j loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(1,1);

    //m loop
    loops[2].set_subspace_looped(0,2);
    loops[2].set_subspace_looped(2,2);

    //k loop
    loops[3].set_subspace_looped(1,0);
    loops[3].set_subspace_looped(2,0);

    //l loop
    loops[4].set_subspace_looped(1,2);
    loops[4].set_subspace_looped(2,1);

    sparse_loop_list sll(loops);

    bool threw_exception = false;
    try
    {
    	block_contract2_kernel<double> bc2k(sll);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when matmul-incompatible order specified");
    }
}

//C_ijm = \sum_kl A_klji B_klm
//Invalid because j is in wrong position for matmul
//dimensions: i = 2,j = 3,k = 4,l = 5,m = 6
void block_kernels_test::test_block_contract2_kernel_non_matmul_A_trans_klji() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_contract2_kernel_non_matmul_A_trans_klji()";

    //Just need dummy bispaces for this test
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);
    sparse_bispace<1> spb_l(5);
    sparse_bispace<1> spb_m(6);

    vector< sparse_bispace_any_order > bispaces(1,spb_i|spb_j|spb_m);
    bispaces.push_back(spb_k|spb_l|spb_j|spb_i);
    bispaces.push_back(spb_k|spb_l|spb_m);

    vector<block_loop> loops(5,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,3);
    //j loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(1,2);
    //m loop
    loops[2].set_subspace_looped(0,2);
    loops[2].set_subspace_looped(2,2);
    //k loop
    loops[3].set_subspace_looped(1,0);
    loops[3].set_subspace_looped(2,0);
    //l loop
    loops[4].set_subspace_looped(1,1);
    loops[4].set_subspace_looped(2,1);

    sparse_loop_list sll(loops);

    bool threw_exception = false;
    try
    {
    	block_contract2_kernel<double> bc2k(sll);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when matmul-incompatible order specified");
    }
}

//C_il = \sum_jk A_ijk B_jlk
//Invalid because contracted indices are not all at the beginning of B
//dimensions: i = 2,j = 3,k = 4,l = 5
void block_kernels_test::test_block_contract2_kernel_non_matmul_B_jlk() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_contract2_kernel_non_matmul_B_jlk()";

    //Just need dummy bispaces for this test
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);
    sparse_bispace<1> spb_l(5);

    vector< sparse_bispace_any_order > bispaces(1,spb_i|spb_l);
    bispaces.push_back(spb_i|spb_j|spb_k);
    bispaces.push_back(spb_j|spb_l|spb_k);

    vector<block_loop> loops(4,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,0);
    //l loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(2,1);
    //j loop
    loops[2].set_subspace_looped(1,1);
    loops[2].set_subspace_looped(2,0);
    //k loop
    loops[3].set_subspace_looped(1,2);
    loops[3].set_subspace_looped(2,2);

    sparse_loop_list sll(loops);

    bool threw_exception = false;
    try
    {
    	block_contract2_kernel<double> bc2k(sll);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when matmul-incompatible order specified");
    }
}

//C_ilm = \sum_jk A_ijk B_jkml
//Invalid because uncontracted indices in B are in the wrong order
//dimensions: i = 2,j = 3,k = 4,l = 5
void block_kernels_test::test_block_contract2_kernel_non_matmul_B_jkml() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_contract2_kernel_non_matmul_B_jkml()";

    //Just need dummy bispaces for this test
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);
    sparse_bispace<1> spb_l(5);
    sparse_bispace<1> spb_m(6);

    vector< sparse_bispace_any_order > bispaces(1,spb_i|spb_l|spb_m);
    bispaces.push_back(spb_i|spb_j|spb_k);
    bispaces.push_back(spb_j|spb_k|spb_m|spb_l);

    vector<block_loop> loops(5,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,0);
    //l loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(2,3);
    //m loop
    loops[2].set_subspace_looped(0,2);
    loops[2].set_subspace_looped(2,2);
    //j loop
    loops[3].set_subspace_looped(1,1);
    loops[3].set_subspace_looped(2,0);
    //k loop
    loops[4].set_subspace_looped(1,2);
    loops[4].set_subspace_looped(2,1);

    sparse_loop_list sll(loops);

    bool threw_exception = false;
    try
    {
    	block_contract2_kernel<double> bc2k(sll);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when matmul-incompatible order specified");
    }
}

//C_ikl = \sum_j A_ij B_kjl
//Invalid because contracted indices are not all at the end of B
//dimensions: i = 2,j = 3,k = 4,l = 5
void block_kernels_test::test_block_contract2_kernel_non_matmul_B_trans_kjl() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_contract2_kernel_non_matmul_B_trans_kjl()";

    //Just need dummy bispaces for this test
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);
    sparse_bispace<1> spb_l(5);

    vector< sparse_bispace_any_order > bispaces(1,spb_i|spb_k|spb_l);
    bispaces.push_back(spb_i|spb_j);
    bispaces.push_back(spb_k|spb_j|spb_l);

    vector<block_loop> loops(4,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,0);
    //k loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(2,0);
    //l loop
    loops[2].set_subspace_looped(0,2);
    loops[2].set_subspace_looped(2,2);
    //j loop
    loops[3].set_subspace_looped(1,1);
    loops[3].set_subspace_looped(2,1);

    sparse_loop_list sll(loops);

    bool threw_exception = false;
    try
    {
    	block_contract2_kernel<double> bc2k(sll);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when matmul-incompatible order specified");
    }
}

//C_ikl = \sum_j A_ij B_lkj
//Invalid because uncontracted indices in B are in the wrong order
//dimensions: i = 2,j = 3,k = 4,l = 5
void block_kernels_test::test_block_contract2_kernel_non_matmul_B_trans_lkj() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_contract2_kernel_non_matmul_B_trans_lkj()";

    //Just need dummy bispaces for this test
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);
    sparse_bispace<1> spb_l(5);

    vector< sparse_bispace_any_order > bispaces(1,spb_i|spb_k|spb_l);
    bispaces.push_back(spb_i|spb_j);
    bispaces.push_back(spb_l|spb_k|spb_j);

    vector<block_loop> loops(4,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,0);
    //k loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(2,1);
    //l loop
    loops[2].set_subspace_looped(0,2);
    loops[2].set_subspace_looped(2,0);
    //j loop
    loops[3].set_subspace_looped(1,1);
    loops[3].set_subspace_looped(2,2);

    sparse_loop_list sll(loops);

    bool threw_exception = false;
    try
    {
    	block_contract2_kernel<double> bc2k(sll);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when matmul-incompatible order specified");
    }
}

//Should throw an exception due to passing a dimension list without the right number of dimensions
//for each bispace
void block_kernels_test::test_block_contract2_kernel_2d_not_enough_dims_and_ptrs() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_not_enough_dims()";

    //C_ji = \sum_k A_ik B_kj
    //dimensions: i = 2,j = 3,k = 4
    //Just need dummy bispaces for this test
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);

    vector< sparse_bispace_any_order > bispaces(1,spb_i | spb_j);
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

	block_contract2_kernel<double> bc2k(sll);

	//Fake dimensions, just care about how many are bassed
	dim_list dl(2,2);
	//DELIBERATELY OMIT dl_B
	vector<dim_list> dim_lists(2,dl);

	vector<double*> ptrs(3,NULL);

	//Check dims
    bool threw_exception = false;
    try
    {
    	bc2k(ptrs,dim_lists);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when not enough dim_lists passed");
    }

    //Check ptrs by fixing dim_lists and breaking ptrs
    ptrs.pop_back();
    dim_lists.push_back(dim_lists.back());
    threw_exception = false;
    try
    {
    	bc2k(ptrs,dim_lists);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when not enough ptrs passed");
    }

}

//Should throw exception when length of dim_lists doesn't match order of bispaces
void block_kernels_test::test_block_contract2_kernel_2d_invalid_dims() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_invalid_dims()";

    //C_ji = \sum_k A_ik B_kj
    //dimensions: i = 2,j = 3,k = 4
    //Just need dummy bispaces for this test
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

	block_contract2_kernel<double> bc2k(sll);

	//Fake dimensions, just care about vector length
	dim_list dl(2,2);
	dim_list dl_fail(1,2);
	vector<dim_list> dim_lists;
	dim_lists.push_back(dl);
	//DELIBERATELY MAKE 'A' dim list wrong length
	dim_lists.push_back(dl_fail);
	dim_lists.push_back(dl);
	vector<double*> ptrs(3,NULL);

    bool threw_exception = false;
    try
    {
    	bc2k(ptrs,dim_lists);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when dim_lists of invalid length passed");
    }
}

//Should throw exception when dimensions in dim_lists don't match up with their contraction partners
void block_kernels_test::test_block_contract2_kernel_2d_incompatible_dims() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_incompatible_dims()";

    //C_ji = \sum_k A_ik B_kj
    //dimensions: i = 2,j = 3,k = 4
    //Just need dummy bispaces for this test
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);

    vector< sparse_bispace_any_order > bispaces(1,spb_i | spb_j);
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

	block_contract2_kernel<double> bc2k(sll);

	dim_list dl(2,2);
	vector<dim_list> dim_lists(3,dl);
	//DELIBERATELY MAKE 'A' dim_list have wrong value for second entry
	dim_lists[1][1] = 3;
	vector<double*> ptrs(3,NULL);

    bool threw_exception = false;
    try
    {
    	bc2k(ptrs,dim_lists);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel()(...) did not throw exception when incompatible dim_lists passed");
    }
}

//Should produce correct matrix multiply output, with matrices in standard order
void block_kernels_test::test_block_contract2_kernel_2d_ik_kj() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_ik_kj()";

    //C_ij = \sum_k A_ik B_kj
    //dimensions: i = 2,j = 3,k = 4
    
    //C
    double test_output_block[6] = {0}; 
    //A
    double test_input_block_1[8] = {1,2,3,4,
                                    5,6,7,8,
                                    };
    //B
    double test_input_block_2[12] = {9,10,11,
                                     12,13,14,
                                     15,16,17,
                                     18,19,20};

    //Correct C
    double correct_output_block[6] = {150, 160, 170, 
                                      366, 392, 418};
    
    vector<double*> ptrs(1,test_output_block);
    ptrs.push_back(test_input_block_1);
    ptrs.push_back(test_input_block_2);

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

    block_contract2_kernel<double> bc2k(sll);

    size_t C_dim_list_arr[2] = {2,3};
    size_t A_dim_list_arr[2] = {2,4};
    size_t B_dim_list_arr[2] = {4,3};

    vector<dim_list> dim_lists(3,dim_list(2));
    for(size_t i = 0; i < 2; ++i) dim_lists[0][i] = C_dim_list_arr[i];
    for(size_t i = 0; i < 2; ++i) dim_lists[1][i] = A_dim_list_arr[i];
    for(size_t i = 0; i < 2; ++i) dim_lists[2][i] = B_dim_list_arr[i];

    bc2k(ptrs,dim_lists);

    for(int i = 0; i < 6; ++i)
    {
        if(test_output_block[i] != correct_output_block[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_contract2_kernel::operator(...) did not produce correct result");
        }
    }
}

//Should produce correct matrix multiply output, with b transposed (both matrices have same fast index
void block_kernels_test::test_block_contract2_kernel_2d_ik_jk() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_ik_jk()";

    //C_ij = \sum_k A_ik B_jk
    //dimensions: i = 2,j = 3,k = 4
    //C
    double test_output_block[6] = {0}; 
    //A
    double test_input_block_1[8] = {1,2,3,4,
                                    5,6,7,8,
                                    };
    //B
    double test_input_block_2[12] = {9,12,15,18,
                                     10,13,16,19,
                                     11,14,17,20};

    //Correct C
    double correct_output_block[6] = {150, 160, 170, 
                                      366, 392, 418};

    vector<double*> ptrs(1,test_output_block);
    ptrs.push_back(test_input_block_1);
    ptrs.push_back(test_input_block_2);

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

    block_contract2_kernel<double> bc2k(sll);

    size_t C_dim_list_arr[2] = {2,3};
    size_t A_dim_list_arr[2] = {2,4};
    size_t B_dim_list_arr[2] = {3,4};

    vector<dim_list> dim_lists(3,dim_list(2));
    for(size_t i = 0; i < 2; ++i) dim_lists[0][i] = C_dim_list_arr[i];
    for(size_t i = 0; i < 2; ++i) dim_lists[1][i] = A_dim_list_arr[i];
    for(size_t i = 0; i < 2; ++i) dim_lists[2][i] = B_dim_list_arr[i];

    bc2k(ptrs,dim_lists);

    for(int i = 0; i < 6; ++i)
    {
        if(test_output_block[i] != correct_output_block[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_contract2_kernel::operator(...) did not produce correct result");
        }
    }
}

void block_kernels_test::test_block_contract2_kernel_2d_ki_kj() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_ki_kj()";

    //C_ij = \sum_k A_ki B_kj
    //dimensions: i = 2,j = 3,k = 4
    
    //C
    double test_output_block[6] = {0}; 
    //A
    double test_input_block_1[8] = {1,5,
                                    2,6,
                                    3,7,
                                    4,8};

    //B
    double test_input_block_2[12] = {9,10,11,
                                     12,13,14,
                                     15,16,17,
                                     18,19,20};

    //Correct C
    double correct_output_block[6] = {150, 160, 170, 
                                      366, 392, 418};

    vector<double*> ptrs(1,test_output_block);
    ptrs.push_back(test_input_block_1);
    ptrs.push_back(test_input_block_2);

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


    block_contract2_kernel<double> bc2k(sll);

    size_t C_dim_list_arr[2] = {2,3};
    size_t A_dim_list_arr[2] = {4,2};
    size_t B_dim_list_arr[2] = {4,3};

    vector<dim_list> dim_lists(3,dim_list(2));
    for(size_t i = 0; i < 2; ++i) dim_lists[0][i] = C_dim_list_arr[i];
    for(size_t i = 0; i < 2; ++i) dim_lists[1][i] = A_dim_list_arr[i];
    for(size_t i = 0; i < 2; ++i) dim_lists[2][i] = B_dim_list_arr[i];

    bc2k(ptrs,dim_lists);
    
    for(int i = 0; i < 6; ++i)
    {
        if(test_output_block[i] != correct_output_block[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_contract2_kernel::operator(...) did not produce correct result");
        }
    }
}

void block_kernels_test::test_block_contract2_kernel_2d_ki_jk() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_ki_jk()";

    //C_ij = \sum_k A_ki B_jk
    //dimensions: i = 2,j = 3,k = 4
    
    //C
    double test_output_block[6] = {0}; 
    //A
    double test_input_block_1[8] = {1,5,
                                    2,6,
                                    3,7,
                                    4,8};

    double test_input_block_2[12] = {9,12,15,18,
                                     10,13,16,19,
                                     11,14,17,20};

    //Correct C
    double correct_output_block[6] = {150, 160, 170, 
                                      366, 392, 418};

    vector<double*> ptrs(1,test_output_block);
    ptrs.push_back(test_input_block_1);
    ptrs.push_back(test_input_block_2);

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

    block_contract2_kernel<double> bc2k(sll);

    size_t C_dim_list_arr[2] = {2,3};
    size_t A_dim_list_arr[2] = {4,2};
    size_t B_dim_list_arr[2] = {3,4};

    vector<dim_list> dim_lists(3,dim_list(2));
    for(size_t i = 0; i < 2; ++i) dim_lists[0][i] = C_dim_list_arr[i];
    for(size_t i = 0; i < 2; ++i) dim_lists[1][i] = A_dim_list_arr[i];
    for(size_t i = 0; i < 2; ++i) dim_lists[2][i] = B_dim_list_arr[i];

    bc2k(ptrs,dim_lists);


    for(int i = 0; i < 6; ++i)
    {
        if(test_output_block[i] != correct_output_block[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_contract2_kernel::operator(...) did not produce correct result");
        }
    }
}

//This test ensures that the contraction still works when the block loops are in a nontraditional order, 
//as long as the subspaces are compatible
void block_kernels_test::test_block_contract2_kernel_2d_ki_kj_permuted_loops() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_ki_kj_permuted_loops()";

    //C_ij = \sum_k A_ki B_kj
    //dimensions: i = 2,j = 3,k = 4
    
    //C
    double test_output_block[6] = {0}; 
    //A
    double test_input_block_1[8] = {1,5,
                                    2,6,
                                    3,7,
                                    4,8};

    //B
    double test_input_block_2[12] = {9,10,11,
                                     12,13,14,
                                     15,16,17,
                                     18,19,20};

    //Correct C
    double correct_output_block[6] = {150, 160, 170, 
                                      366, 392, 418};

    vector<double*> ptrs(1,test_output_block);
    ptrs.push_back(test_input_block_1);
    ptrs.push_back(test_input_block_2);

    //bispaces
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);

    vector<sparse_bispace_any_order> bispaces(1,spb_i | spb_j);
    bispaces.push_back(spb_k|spb_i);
    bispaces.push_back(spb_k|spb_j);

    vector<block_loop> loops(3,block_loop(bispaces));

    //k loop
    loops[0].set_subspace_looped(1,0);
    loops[0].set_subspace_looped(2,0);

    //i loop
    loops[1].set_subspace_looped(0,0);
    loops[1].set_subspace_looped(1,1);
    //j loop
    loops[2].set_subspace_looped(0,1);
    loops[2].set_subspace_looped(2,1);

    sparse_loop_list sll(loops);


    block_contract2_kernel<double> bc2k(sll);

    size_t C_dim_list_arr[2] = {2,3};
    size_t A_dim_list_arr[2] = {4,2};
    size_t B_dim_list_arr[2] = {4,3};

    vector<dim_list> dim_lists(3,dim_list(2));
    for(size_t i = 0; i < 2; ++i) dim_lists[0][i] = C_dim_list_arr[i];
    for(size_t i = 0; i < 2; ++i) dim_lists[1][i] = A_dim_list_arr[i];
    for(size_t i = 0; i < 2; ++i) dim_lists[2][i] = B_dim_list_arr[i];

    bc2k(ptrs,dim_lists);
    
    for(int i = 0; i < 6; ++i)
    {
        if(test_output_block[i] != correct_output_block[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_contract2_kernel::operator(...) did not produce correct result");
        }
    }
}

//Contract 3d with 2d
void block_kernels_test::test_block_contract2_kernel_3d_2d() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_contract2_kernel_3d_2d()";

    //C_ijl = \sum_k A_ijk B_kl
    //dimensions: i = 2,j = 3,k = 4,l=5
    
    //C
    double test_output_block[30] = {0}; 

    //A
    double test_input_block_1[24] = {//i = 0
                                     1,2,3,4,
                                     5,6,7,8,
                                     9,10,11,12,

                                     //i = 1
                                     13,14,15,16,
                                     17,18,19,20,
                                     21,22,23,24};

    //B 
    double test_input_block_2[20] = {1,2,3,4,5,
                                     6,7,8,9,10,
                                     11,12,13,14,15,
                                     16,17,18,19,20};

    //Correct C
    double correct_output_block[30] = {//i = 0
                                       110,120,130,140,150, 
                                       246,272,298,324,350,
                                       382,424,466,508,550,

                                       //i = 1
                                       518,576,634,692,750,
                                       654,728,802,876,950,
                                       790,880,970,1060,1150};

    vector<double*> ptrs(1,test_output_block);
    ptrs.push_back(test_input_block_1);
    ptrs.push_back(test_input_block_2);

    //bispaces
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);
    sparse_bispace<1> spb_l(5);

    vector< sparse_bispace_any_order > bispaces(1,spb_i|spb_j|spb_l);
    bispaces.push_back(spb_i|spb_j|spb_k);
    bispaces.push_back(spb_k|spb_l);

    vector<block_loop> loops(4,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,0);
    //j loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(1,1);
    //l loop
    loops[2].set_subspace_looped(0,2);
    loops[2].set_subspace_looped(2,1);
    //k loop
    loops[3].set_subspace_looped(1,2);
    loops[3].set_subspace_looped(2,0);

    sparse_loop_list sll(loops);

    block_contract2_kernel<double> bc2k(sll);

    size_t C_dim_list_arr[3] = {2,3,5};
    size_t A_dim_list_arr[3] = {2,3,4};
    size_t B_dim_list_arr[2] = {4,5};

    vector<dim_list> dim_lists(2,dim_list(3));
    dim_lists.push_back(dim_list(2));
    for(size_t i = 0; i < 3; ++i) dim_lists[0][i] = C_dim_list_arr[i];
    for(size_t i = 0; i < 3; ++i) dim_lists[1][i] = A_dim_list_arr[i];
    for(size_t i = 0; i < 2; ++i) dim_lists[2][i] = B_dim_list_arr[i];

    bc2k(ptrs,dim_lists);

    for(int i = 0; i < 30; ++i)
    {
        if(test_output_block[i] != correct_output_block[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_contract2_kernel::operator(...) did not produce correct result");
        }
    }
}

void block_kernels_test::test_block_contract2_kernel_3d_3d_multi_index() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_contract2_kernel_3d_3d_multi_index()";

    //Cil = Aijk Bljk
    //2x5 2x3x4 5x3x4
    double A_arr[24] = { //i = 0
                         1,2,3,4,
                         5,6,7,8,
                         9,10,11,12,

                         //i = 1
                         13,14,15,16,
                         17,18,19,20,
                         21,22,23,24};

    double B_arr[60] = { //l = 0
                         1,2,3,4,
                         5,6,7,8,
                         9,10,11,12,

                         //l = 1
                         13,14,15,16,
                         17,18,19,20,
                         21,22,23,24,

                         //l = 2
                         25,26,27,28,
                         29,30,31,32,
                         33,34,35,36,

                         //l = 3
                         37,38,39,40,
                         41,42,43,44,
                         45,46,47,48,

                         //l = 4
                         49,50,51,52,
                         53,54,55,56,
                         57,58,59,60};

    double C_arr[10] = {0};

    double C_correct_arr[10] = { //i = 0
                                 650,1586,2522,3458,4394,
                                 //i = 1
                                 1586,4250,6914,9578,12242};

    vector<double*> ptrs(1,C_arr);
    ptrs.push_back(A_arr);
    ptrs.push_back(B_arr);

    //bispaces
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);
    sparse_bispace<1> spb_k(4);
    sparse_bispace<1> spb_l(5);

    vector< sparse_bispace_any_order > bispaces(1,spb_i|spb_l);
    bispaces.push_back(spb_i|spb_j|spb_k);
    bispaces.push_back(spb_l|spb_j|spb_k);

    vector<block_loop> loops(4,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(1,0);
    //l loop
    loops[1].set_subspace_looped(0,1);
    loops[1].set_subspace_looped(2,0);
    //j loop
    loops[2].set_subspace_looped(1,1);
    loops[2].set_subspace_looped(2,1);
    //k loop
    loops[3].set_subspace_looped(1,2);
    loops[3].set_subspace_looped(2,2);

    sparse_loop_list sll(loops);

    block_contract2_kernel<double> bc2k(sll);

    size_t C_dim_list_arr[2] = {2,5};
    size_t A_dim_list_arr[3] = {2,3,4};
    size_t B_dim_list_arr[3] = {5,3,4};

    vector<dim_list> dim_lists(1,dim_list(2));
    dim_lists.push_back(dim_list(3));
    dim_lists.push_back(dim_list(3));
    for(size_t i = 0; i < 2; ++i) dim_lists[0][i] = C_dim_list_arr[i];
    for(size_t i = 0; i < 3; ++i) dim_lists[1][i] = A_dim_list_arr[i];
    for(size_t i = 0; i < 3; ++i) dim_lists[2][i] = B_dim_list_arr[i];

    bc2k(ptrs,dim_lists);

    for(int i = 0; i < 10; ++i)
    {
        if(C_arr[i] != C_correct_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_contract2_kernel::operator(...) did not produce correct result");
        }
    }
}

void block_kernels_test::test_block_contract2_kernel_matrix_vector_mult() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_contract2_kernel_matrix_vector_mult()";

    double x_arr[3] = {1,2,3}; 

    double A_arr[6] = {1,2,3,
                       4,5,6};

    double C_arr[2] = {0};

    double C_correct_arr[2] = {14,32};
                                 

    vector<double*> ptrs(1,C_arr);
    ptrs.push_back(x_arr);
    ptrs.push_back(A_arr);

    //bispaces
    sparse_bispace<1> spb_i(2);
    sparse_bispace<1> spb_j(3);

    vector< sparse_bispace_any_order > bispaces(1,spb_i);
    bispaces.push_back(spb_j);
    bispaces.push_back(spb_i|spb_j);

    vector<block_loop> loops(2,block_loop(bispaces));
    //i loop
    loops[0].set_subspace_looped(0,0);
    loops[0].set_subspace_looped(2,0);
    //j loop
    loops[1].set_subspace_looped(1,0);
    loops[1].set_subspace_looped(2,1);

    sparse_loop_list sll(loops);

    block_contract2_kernel<double> bc2k(sll);

    size_t C_dim_list_arr[1] = {2};
    size_t x_dim_list_arr[1] = {3};
    size_t A_dim_list_arr[2] = {2,3};

    vector<dim_list> dim_lists(1,dim_list(1));
    dim_lists.push_back(dim_list(1));
    dim_lists.push_back(dim_list(2));
    for(size_t i = 0; i < 1; ++i) dim_lists[0][i] = C_dim_list_arr[i];
    for(size_t i = 0; i < 1; ++i) dim_lists[1][i] = x_dim_list_arr[i];
    for(size_t i = 0; i < 2; ++i) dim_lists[2][i] = A_dim_list_arr[i];

    bc2k(ptrs,dim_lists);

    for(int i = 0; i < 2; ++i)
    {
        if(C_arr[i] != C_correct_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_contract2_kernel::operator(...) did not produce correct result");
        }
    }
}

void block_kernels_test::test_block_subtract2_kernel_not_enough_dims_and_ptrs() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_subtract2_kernel_not_enough_dims_and_ptrs()";

    //Just need dummy data for this test
    vector<double*> ptrs(3);

    dim_list C_dims(1,3);
    C_dims.push_back(4);
    dim_list A_dims(1,3);
    A_dims.push_back(4);
    //Intentionally leave off B dims

    vector<dim_list> dim_lists(1,C_dims);
    dim_lists.push_back(A_dims);

    //Should throw exception for not enough dim_lists
    block_subtract2_kernel<double> bs2k;
    bool threw_exception = false;
    try
    {
    	bs2k(ptrs,dim_lists);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_subtract2_kernel::operator(...) did not throw exception when not enough dim_lists passed");
    }
}

//Should throw exception when dim_lists are not equal in size or values
void block_kernels_test::test_block_subtract2_kernel_invalid_dims() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_subtract2_kernel_invalid_dims()";

    //Just need dummy data for this test
    vector<double*> ptrs(3);

    dim_list C_dims(1,3);
    C_dims.push_back(4);
    dim_list A_dims(1,3);
    A_dims.push_back(4);
    //Intentionally leave off 2nd dim of B
    dim_list B_dims(1,3);

    vector<dim_list> dim_lists(1,C_dims);
    dim_lists.push_back(A_dims);
    dim_lists.push_back(B_dims);

    //First check incompatible sizes
    block_subtract2_kernel<double> bs2k;
    bool threw_exception = false;
    try
    {
    	bs2k(ptrs,dim_lists);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_subtract2_kernel::operator(...) did not throw exception when not enough dimensions for one block passed");
    }

    //Now check incompatible dimensions
    threw_exception = false;
    dim_lists[2].push_back(5);
    try
    {
    	bs2k(ptrs,dim_lists);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }
    if(!threw_exception)
    {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_subtract2_kernel::operator(...) did not throw exception when wrong dimensions for one block passed");
    }

}

void block_kernels_test::test_block_subtract2_kernel_2d() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_subtract2_kernel_2d()";

    double C[12];
    double A[12] = {3,7,9,4,
    				2,8,14,11,
    				12,20,15,8};
    double B[12] = {1,2,3,4,
    				5,6,7,8,
    				9,10,11,12};
    double C_correct[12] = {2,5,6,0,
    						-3,2,7,3,
    						3,10,4,-4};

    //Just need dummy data for this test
    vector<double*> ptrs(1,C);
    ptrs.push_back(A);
    ptrs.push_back(B);

    dim_list C_dims(1,3);
    C_dims.push_back(4);
    dim_list A_dims(1,3);
    A_dims.push_back(4);
    dim_list B_dims(1,3);
    B_dims.push_back(4);

    vector<dim_list> dim_lists(1,C_dims);
    dim_lists.push_back(A_dims);
    dim_lists.push_back(B_dims);

    //First check incompatible sizes
    block_subtract2_kernel<double> bs2k;
	bs2k(ptrs,dim_lists);
	for(size_t i = 0; i < 12; ++i)
	{
		if(C[i] != C_correct[i])
		{
			fail_test(test_name,__FILE__,__LINE__,
					"block_subtract2_kernel::operator(...) returned incorrect output");
		}
	}
}

} // namespace libtensor
