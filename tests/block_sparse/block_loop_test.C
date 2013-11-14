//TODO: remove block_loop.h
#include <libtensor/block_sparse/block_loop.h>
#include <libtensor/block_sparse/block_loop_new.h>
#include <libtensor/block_sparse/block_kernels.h>
#include <sstream>
#include "block_loop_test.h" 

namespace libtensor {

void block_loop_test::perform() throw(libtest::test_exception) {

    test_set_subspace_looped_invalid_bispace_idx();
    test_set_subspace_looped_invalid_subspace_idx();
    test_set_subspace_looped_not_matching_subspaces();

    test_get_subspace_looped_invalid_bispace_idx();
    test_get_subspace_looped();

    test_is_bispace_ignored_invalid_bispace_idx();
    test_is_bispace_ignored();

    /*
     * OLD TESTS REFACTOR
     */
    test_run_invalid_bispaces();

    test_run_block_permute_kernel_2d();
    test_run_block_permute_kernel_2d_sparse();
    test_run_block_permute_kernel_3d_201();
    test_run_block_permute_kernel_3d_201_sparse();


    test_run_block_contract2_kernel_2d_2d();
    test_run_block_contract2_kernel_3d_2d();
}

void block_loop_test::test_set_subspace_looped_invalid_bispace_idx()
		throw (libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_set_subspace_looped_invalid_bispace_idx()";

	//bispaces
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    std::vector< sparse_bispace_any_order > bispaces(2,spb_1|spb_2);
    block_loop_new bl(bispaces);

    //Fails because there is no third bispace
    bool threw_exception = false;
    try
    {
		bl.set_subspace_looped(3,1);
    }
    catch(out_of_bounds&)
    {
    	threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::set_subspace_looped(...) did not throw exception when invalid bispace index specified");
    }
}

void block_loop_test::test_set_subspace_looped_invalid_subspace_idx()
		throw (libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_set_subspace_looped_invalid_bispace_idx()";

	//bispaces
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    std::vector< sparse_bispace_any_order > bispaces(2,spb_1|spb_2);

    block_loop_new bl(bispaces);

    //Fails because 2nd subspace has no 3rd subspace
    bool threw_exception = false;
    try
    {
		bl.set_subspace_looped(1,2);
    }
    catch(out_of_bounds&)
    {
    	threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::set_subspace_looped(...) did not throw exception when invalid subspace index specified");
    }
}

void block_loop_test::test_set_subspace_looped_not_matching_subspaces()
		throw (libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_set_subspace_looped_not_matching_subspaces()";

	//bispaces
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    std::vector< sparse_bispace_any_order > bispaces(2,spb_1|spb_2);

    block_loop_new bl(bispaces);

    //Dimension 4
    bl.set_subspace_looped(0,0);
    //Fails because loop is accessing incompatible subspaces of the two bispaces
    bool threw_exception = false;
    try
    {
    	//Dimension 5
		bl.set_subspace_looped(1,1);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::set_subspace_looped(...) did not throw exception when two incompatible subspace indices specified");
    }
}

void block_loop_test::test_get_subspace_looped_invalid_bispace_idx()
		throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_get_subspace_looped_invalid_bispace_idx()";

	//bispaces
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    sparse_bispace<1> spb_3(6);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    split_points_3.push_back(4);
    spb_3.split(split_points_3);

    std::vector< sparse_bispace_any_order > bispaces;
    bispaces.push_back(spb_1|spb_2|spb_3);
    bispaces.push_back(spb_2|spb_3);
    bispaces.push_back(spb_2);

    //This loop ignores the second bispace
    block_loop_new bl(bispaces);
    bl.set_subspace_looped(0,1);
    bl.set_subspace_looped(2,0);

    //Out of bounds, only 3 bispaces, should fail
    bool threw_exception = false;
    try
    {
    	bl.get_subspace_looped(3);
    }
    catch(out_of_bounds&)
    {
    	threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::get_subspace_looped(...) did not throw exception when bispace requested out of bounds");
    }

    //Second bispace is not looped, should throw exception
    threw_exception = false;
    try
    {
    	bl.get_subspace_looped(1);
    }
    catch(bad_parameter&)
    {
    	threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::get_subspace_looped(...) did not throw exception when bispace requested not looped over");
    }
}

void block_loop_test::test_get_subspace_looped()
		throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_get_subspace_looped()";

	//bispaces
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    sparse_bispace<1> spb_3(6);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    split_points_3.push_back(4);
    spb_3.split(split_points_3);

    std::vector< sparse_bispace_any_order > bispaces;
    bispaces.push_back(spb_1|spb_2|spb_3);
    bispaces.push_back(spb_2|spb_3);
    bispaces.push_back(spb_2);

    //This loop ignores the second bispace
    block_loop_new bl(bispaces);
    bl.set_subspace_looped(0,1);
    bl.set_subspace_looped(2,0);

    if(bl.get_subspace_looped(0) != 1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::get_subspace_looped(...) returned incorrect value");
    }
}

void block_loop_test::test_is_bispace_ignored_invalid_bispace_idx()
		throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_is_bispace_ignored_invalid_bispace_idx()";

	//bispaces
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    sparse_bispace<1> spb_3(6);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    split_points_3.push_back(4);
    spb_3.split(split_points_3);

    std::vector< sparse_bispace_any_order > bispaces;
    bispaces.push_back(spb_1|spb_2|spb_3);
    bispaces.push_back(spb_2|spb_3);
    bispaces.push_back(spb_2);

    //This loop ignores the second bispace
    block_loop_new bl(bispaces);
    bl.set_subspace_looped(0,1);
    bl.set_subspace_looped(2,0);

    //Only three bispaces should fail
    bool threw_exception = false;
    try
    {
    	bl.is_bispace_ignored(3);
    }
    catch(out_of_bounds)
    {
    	threw_exception = true;
    }

    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::is_bispace_ignored(...) did not throw exception when bispace index out of bounds");
    }
}

void block_loop_test::test_is_bispace_ignored()
		throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_is_bispace_ignored()";

	//bispaces
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    sparse_bispace<1> spb_3(6);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    split_points_3.push_back(4);
    spb_3.split(split_points_3);

    std::vector< sparse_bispace_any_order > bispaces;
    bispaces.push_back(spb_1|spb_2|spb_3);
    bispaces.push_back(spb_2|spb_3);
    bispaces.push_back(spb_2);

    //This loop ignores the second bispace
    block_loop_new bl(bispaces);
    bl.set_subspace_looped(0,1);
    bl.set_subspace_looped(2,0);

    if(!bl.is_bispace_ignored(1))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::is_bispace_ignored(...) did not return true for ignored bispace");
    }

    if(bl.is_bispace_ignored(0))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_loop::is_bispace_ignored(...) did returned true for non-ignored bispace");
    }
}

//Should throw an exception if the bispaces passed to run() do not match as specified in the construction of the loops
void block_loop_test::test_run_invalid_bispaces() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_run_invalid_bispace()";

    //Dummy data suffices for this test
    double test_input_arr[20];
    double test_output_arr[20];

    //First bispace (slow index in input) and splitting
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    //Second bispace (fast index in input) and splitting
    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    sparse_bispace<2> two_d = spb_1 | spb_2;

    runtime_permutation perm(2);
    perm.permute(0,1);
    block_permute_kernel<double> bpk(perm);


    sequence<1,size_t> output_bispace_indices_1(0);
    sequence<1,size_t> input_bispace_indices_1(1);
    sequence<1,bool> output_ignore_1(false);
    sequence<1,bool> input_ignore_1(false);

    sequence<1,size_t> output_bispace_indices_2(1);
    sequence<1,size_t> input_bispace_indices_2(0);
    sequence<1,bool> output_ignore_2(false);
    sequence<1,bool> input_ignore_2(false);


    std::vector< block_loop<1,1> > loop_list;
	loop_list.push_back(block_loop<1,1>(output_bispace_indices_1,
                                        input_bispace_indices_1,
                                        output_ignore_1,
                                        input_ignore_1));

    loop_list.push_back(block_loop<1,1>(output_bispace_indices_1,
                        input_bispace_indices_1,
                        output_ignore_1,
                        input_ignore_1));


    sequence<1,double*> output_ptrs(test_output_arr); 
    sequence<1,const double*> input_ptrs(test_input_arr); 

    //Here, one of the bispaces should be transposed but is not... this should cause an exception because the bispaces
    //do not line up in the loops
    sequence<1,sparse_bispace_any_order> output_bispaces(two_d);
    sequence<1,sparse_bispace_any_order> input_bispaces(two_d);

    bool threw_exception = false;
    try
    {
        run_loop_list(loop_list,bpk,output_ptrs,input_ptrs,output_bispaces,input_bispaces);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "run_loop_list(...) did not throw exception when passed incompatible bispaces");
    }

}

//Permuted nested loops
void block_loop_test::test_run_block_permute_kernel_2d() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_run_block_permute_kernel_2d()";

	//Indices in comments are block indices
    double test_input_arr[20] = { //i = 0, j = 0
                                  1,2,
                                  6,7,

                                 //i = 0, j = 1
                                 3,4,5,
                                 8,9,10,

                                 //i = 1, j = 0
                                 11,12,
                                 16,17,

                                 //i = 0, j = 1
                                 13,14,15,
                                 18,19,20 };

    double correct_output_arr[20] = { //j = 0, i = 0
                                 	  1,6,
                                      2,7,

									  //j = 0, i = 1
									  11,16,
									  12,17,

									  //j = 1, i = 0
									  3,8,
									  4,9,
									  5,10,

									  //j = 1, i = 1
									  13,18,
									  14,19,
									  15,20 };

    double test_output_arr[20];

    //First bispace (slow index) and splitting
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    //Second bispace (fast index) and splitting
    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    spb_2.split(split_points_2);

    sparse_bispace<2> two_d_input = spb_1 | spb_2;
    sparse_bispace<2> two_d_output = spb_2 | spb_1;

    runtime_permutation perm(2);
    perm.permute(0,1);
    block_permute_kernel<double> bpk(perm);


    sequence<1,size_t> output_bispace_indices_2(0);
    sequence<1,size_t> input_bispace_indices_2(1);
    sequence<1,bool> output_ignore_2(false);
    sequence<1,bool> input_ignore_2(false);

    sequence<1,size_t> output_bispace_indices_1(1);
    sequence<1,size_t> input_bispace_indices_1(0);
    sequence<1,bool> output_ignore_1(false);
    sequence<1,bool> input_ignore_1(false);

    std::vector< block_loop<1,1> > loop_list;
	loop_list.push_back(block_loop<1,1>(output_bispace_indices_2,
						input_bispace_indices_2,
						output_ignore_2,
						input_ignore_2));

    loop_list.push_back(block_loop<1,1>(output_bispace_indices_1,
                        input_bispace_indices_1,
                        output_ignore_1,
                        input_ignore_1));

    sequence<1,double*> output_ptrs(test_output_arr); 
    sequence<1,const double*> input_ptrs(test_input_arr); 
    sequence<1,sparse_bispace_any_order> output_bispaces(two_d_output);
    sequence<1,sparse_bispace_any_order> input_bispaces(two_d_input);

    run_loop_list(loop_list,bpk,output_ptrs,input_ptrs,output_bispaces,input_bispaces);

    for(int i = 0; i < 20; ++i)
    {
        if(test_output_arr[i] != correct_output_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_loop<M,N,T>::run(...) produced incorrect output");
        }
    }

}

//Now we permute a sparse matrix, so the dimension of the arrays is smaller
void block_loop_test::test_run_block_permute_kernel_2d_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_run_block_permute_kernel_2d_sparse()";

	//Indices in comments are block indices
    double test_input_arr[14] = { //i = 0, j = 0
                                  1,2,
                                  3,4,

                                  //i = 0, j = 2
                                  5,6,7,
                                  8,9,10,

                                  //i = 1, j = 1
                                  11,12,
                                  13,14};

    double correct_output_arr[14] = { //j = 0, i = 0
                                      1,3,
                                      2,4,

									  //j = 1, i = 1
                                      11,13,
                                      12,14,

									  //j = 2, i = 0
                                      5,8,
                                      6,9,
                                      7,10};

    double test_output_arr[14];

    //First bispace (slow index) and splitting
    sparse_bispace<1> spb_1(4);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    spb_1.split(split_points_1);

    //Second bispace (fast index) and splitting
    sparse_bispace<1> spb_2(7);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    split_points_2.push_back(4);
    spb_2.split(split_points_2);

    //Sparsity information
    std::vector< sequence<2,size_t> > sig_blocks(3);
    sig_blocks[0][0] = 0;
    sig_blocks[0][1] = 0;
    sig_blocks[1][0] = 0;
    sig_blocks[1][1] = 2;
    sig_blocks[2][0] = 1;
    sig_blocks[2][1] = 1;

    permutation<2> perm;
    perm.permute(0,1);
    sparse_bispace<2> two_d_input = spb_1 % spb_2 << sig_blocks;
    sparse_bispace<2> two_d_output = two_d_input.permute(perm);

    runtime_permutation rperm(2);
    rperm.permute(0,1);
    block_permute_kernel<double> bpk(rperm);


    sequence<1,size_t> output_bispace_indices_2(0);
    sequence<1,size_t> input_bispace_indices_2(1);
    sequence<1,bool> output_ignore_2(false);
    sequence<1,bool> input_ignore_2(false);

    sequence<1,size_t> output_bispace_indices_1(1);
    sequence<1,size_t> input_bispace_indices_1(0);
    sequence<1,bool> output_ignore_1(false);
    sequence<1,bool> input_ignore_1(false);

    std::vector< block_loop<1,1> > loop_list;
	loop_list.push_back(block_loop<1,1>(output_bispace_indices_2,
						input_bispace_indices_2,
						output_ignore_2,
						input_ignore_2));

    loop_list.push_back(block_loop<1,1>(output_bispace_indices_1,
                        input_bispace_indices_1,
                        output_ignore_1,
                        input_ignore_1));

    sequence<1,double*> output_ptrs(test_output_arr); 
    sequence<1,const double*> input_ptrs(test_input_arr); 
    sequence<1,sparse_bispace_any_order> output_bispaces(two_d_output);
    sequence<1,sparse_bispace_any_order> input_bispaces(two_d_input);

    run_loop_list(loop_list,bpk,output_ptrs,input_ptrs,output_bispaces,input_bispaces);

    for(int i = 0; i < 14; ++i)
    {
        if(test_output_arr[i] != correct_output_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_loop<M,N,T>::run(...) produced incorrect output");
        }
    }

}

void block_loop_test::test_run_block_permute_kernel_3d_201() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_run_block_permute_kernel_3d_201()";

    //3x4x5
    //Permutation is kij -> ijk 
	//Indices in comments are block indices
    double test_input_arr[60] = { // k = 0, i = 0; j = 0
                                  1,2,
                                  3,4,
                                  5,6,

                                  // k = 0, i = 0, j = 1                        
                                  7,8,9,
                                  10,11,12,
                                  13,14,15,

                                  // k = 0, i = 1, j = 0
                                  16,17,

                                  // k = 0, i = 1, j = 1
                                  18,19,20,

                                  //k = 1, i = 0, j = 0
                                  21,22,
                                  23,24,
                                  25,26,
                                  27,28,
                                  29,30,
                                  31,32,

                                  //k = 1, i = 0, j = 1
                                  33,34,35,
                                  36,37,38,
                                  39,40,41,
                                  42,43,44,
                                  45,46,47,
                                  48,49,50,

                                  //k = 1, i = 1, j = 0
                                  51,52,
                                  53,54,

                                  //k = 1, i = 1, j = 1
                                  55,56,57,
                                  58,59,60};

    double correct_output_arr[60] = { //i = 0, j = 0, k = 0
                                      1,
                                      2,
                                      3,
                                      4,
                                      5,
                                      6,

                                      //i = 0, j = 0, k = 1
                                      21,27,
                                      22,28,
                                      23,29,
                                      24,30,
                                      25,31,
                                      26,32,

                                      //i = 0, j = 1, k = 0
                                      7,8,9,
                                      10,11,12,
                                      13,14,15,

                                      //i = 0, j = 1, k = 1
                                      33,42,
                                      34,43,
                                      35,44,
                                      36,45,
                                      37,46,
                                      38,47,
                                      39,48,
                                      40,49,
                                      41,50,

                                      //i = 1, j = 0, k = 0
                                      16,17,

                                      //i = 1, j = 0, k = 1
                                      51,53,
                                      52,54,

                                      //i = 1, j = 1, k = 0
                                      18,19,20,

                                      //i = 1, j = 1, k = 1
                                      55,58,
                                      56,59,
                                      57,60};

    double test_output_arr[60];

    //First bispace (slow index in input) and splitting
    sparse_bispace<1> spb_1(3);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(1);
    spb_1.split(split_points_1);

    //Second bispace (mid index in input) and splitting
    sparse_bispace<1> spb_2(4);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(3);
    spb_2.split(split_points_2);

    //Third bispace (fast index in input) and splitting
    sparse_bispace<1> spb_3(5);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    spb_3.split(split_points_3);

    sparse_bispace<3> three_d_input = spb_1 | spb_2 | spb_3;
    sparse_bispace<3> three_d_output = spb_2 | spb_3 | spb_1;

    runtime_permutation perm(3);
    perm.permute(0,2);
    perm.permute(0,1);
    block_permute_kernel<double> bpk(perm);

    sequence<1,size_t> output_bispace_indices_1(0);
    sequence<1,size_t> input_bispace_indices_1(1);
    sequence<1,bool> output_ignore_1(false);
    sequence<1,bool> input_ignore_1(false);


    sequence<1,size_t> output_bispace_indices_2(1);
    sequence<1,size_t> input_bispace_indices_2(2);
    sequence<1,bool> output_ignore_2(false);
    sequence<1,bool> input_ignore_2(false);

    sequence<1,size_t> output_bispace_indices_3(2);
    sequence<1,size_t> input_bispace_indices_3(0);
    sequence<1,bool> output_ignore_3(false);
    sequence<1,bool> input_ignore_3(false);

    std::vector< block_loop<1,1> > loop_list;
	loop_list.push_back(block_loop<1,1>(output_bispace_indices_1,
						input_bispace_indices_1,
						output_ignore_1,
						input_ignore_1));

    loop_list.push_back(block_loop<1,1>(output_bispace_indices_2,
                        input_bispace_indices_2,
                        output_ignore_2,
                        input_ignore_2));

    loop_list.push_back(block_loop<1,1>(output_bispace_indices_3,
                        input_bispace_indices_3,
                        output_ignore_3,
                        input_ignore_3));




    sequence<1,double*> output_ptrs(test_output_arr); 
    sequence<1,const double*> input_ptrs(test_input_arr); 
    sequence<1,sparse_bispace_any_order> output_bispaces(three_d_output);
    sequence<1,sparse_bispace_any_order> input_bispaces(three_d_input);

    run_loop_list(loop_list,bpk,output_ptrs,input_ptrs,output_bispaces,input_bispaces);

    for(int i = 0; i < 60; ++i)
    {
        if(test_output_arr[i] != correct_output_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_loop<M,N,T>::run(...) produced incorrect output");
        }
    }

}

void block_loop_test::test_run_block_permute_kernel_3d_201_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_run_block_permute_kernel_3d_201_sparse()";

    //3x4x5
    //Permutation is kij -> ijk 
	//Indices in comments are block indices
    double test_input_arr[35] = { //k = 0, i = 0; j = 0
                                  1,2,
                                  3,4,
                                  5,6,

                                  //k = 0, i = 0, j = 1                        
                                  7,8,9,
                                  10,11,12,
                                  13,14,15,

                                  //k = 0, i = 1, j = 0
                                  16,17,

                                  //k = 1, i = 0, j = 0
                                  21,22,
                                  23,24,
                                  25,26,
                                  27,28,
                                  29,30,
                                  31,32,

                                  //k = 1, i = 1, j = 1
                                  55,56,57,
                                  58,59,60};

    double correct_output_arr[35] = { //i = 0 j = 0 k = 0
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

                                      //i = 0 j = 1 k = 0
                                      7,
                                      8,
                                      9,
                                      10,
                                      11,
                                      12,
                                      13,
                                      14,
                                      15,

                                      //i = 1 j = 0 k = 0
                                      16,
                                      17,

                                      // i = 1, j = 1 k = 1
                                      55,58,
                                      56,59,
                                      57,60};

    double test_output_arr[35];

    //First bispace (slow index in input) and splitting
    sparse_bispace<1> spb_1(3);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(1);
    spb_1.split(split_points_1);

    //Second bispace (mid index in input) and splitting
    sparse_bispace<1> spb_2(4);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(3);
    spb_2.split(split_points_2);

    //Third bispace (fast index in input) and splitting
    sparse_bispace<1> spb_3(5);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    spb_3.split(split_points_3);

    //Sparsity data
    std::vector< sequence<3,size_t> > sig_blocks(5);
    sig_blocks[0][0] = 0; 
    sig_blocks[0][1] = 0;
    sig_blocks[0][2] = 0;
    sig_blocks[1][0] = 0; 
    sig_blocks[1][1] = 0;
    sig_blocks[1][2] = 1;
    sig_blocks[2][0] = 0; 
    sig_blocks[2][1] = 1;
    sig_blocks[2][2] = 0;
    sig_blocks[3][0] = 1; 
    sig_blocks[3][1] = 0;
    sig_blocks[3][2] = 0;
    sig_blocks[4][0] = 1; 
    sig_blocks[4][1] = 1;
    sig_blocks[4][2] = 1;

    sparse_bispace<3> three_d_input = spb_1 % spb_2 % spb_3 << sig_blocks;
    permutation<3> perm;
    perm.permute(0,2).permute(0,1);
    sparse_bispace<3> three_d_output = three_d_input.permute(perm);

    runtime_permutation rperm(3);
    rperm.permute(0,2);
    rperm.permute(0,1);
    block_permute_kernel<double> bpk(rperm);

    sequence<1,size_t> output_bispace_indices_1(0);
    sequence<1,size_t> input_bispace_indices_1(1);
    sequence<1,bool> output_ignore_1(false);
    sequence<1,bool> input_ignore_1(false);


    sequence<1,size_t> output_bispace_indices_2(1);
    sequence<1,size_t> input_bispace_indices_2(2);
    sequence<1,bool> output_ignore_2(false);
    sequence<1,bool> input_ignore_2(false);

    sequence<1,size_t> output_bispace_indices_3(2);
    sequence<1,size_t> input_bispace_indices_3(0);
    sequence<1,bool> output_ignore_3(false);
    sequence<1,bool> input_ignore_3(false);

    std::vector< block_loop<1,1> > loop_list;
	loop_list.push_back(block_loop<1,1>(output_bispace_indices_1,
						input_bispace_indices_1,
						output_ignore_1,
						input_ignore_1));

    loop_list.push_back(block_loop<1,1>(output_bispace_indices_2,
                        input_bispace_indices_2,
                        output_ignore_2,
                        input_ignore_2));

    loop_list.push_back(block_loop<1,1>(output_bispace_indices_3,
                        input_bispace_indices_3,
                        output_ignore_3,
                        input_ignore_3));




    sequence<1,double*> output_ptrs(test_output_arr); 
    sequence<1,const double*> input_ptrs(test_input_arr); 
    sequence<1,sparse_bispace_any_order> output_bispaces(three_d_output);
    sequence<1,sparse_bispace_any_order> input_bispaces(three_d_input);

    run_loop_list(loop_list,bpk,output_ptrs,input_ptrs,output_bispaces,input_bispaces);

    for(int i = 0; i < 35; ++i)
    {
        if(test_output_arr[i] != correct_output_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_loop<M,N,T>::run(...) produced incorrect output");
        }
    }

}

void block_loop_test::test_run_block_contract2_kernel_2d_2d() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_run_block_contract2_kernel_2d_2d()";

	//Indices in comments are block indices
    //dimensions: i = 4,k = 5,j = 6
    //Contraction takes the form of A * B^t
    double test_input_arr_1[20] = { //i = 0, k = 0
                                  1,2,
                                  6,7,

                                 //i = 0, k = 1
                                 3,4,5,
                                 8,9,10,

                                 //i = 1, k = 0
                                 11,12,
                                 16,17,

                                 //i = 1, k = 1
                                 13,14,15,
                                 18,19,20 };

    double test_input_arr_2[30] = { //j = 0, k = 0 
                                    1,2,
                                    3,4,

                                    //j = 0, k = 1 
                                    5,6,7,
                                    8,9,10,

                                    //j = 1, k = 0
                                    11,12,
                                    13,14,
                                    15,16,
                                    17,18,

                                    //j = 1, k = 1
                                    19,20,21,
                                    22,23,24,
                                    25,26,27,
                                    28,29,30};

    double test_output_arr[24] = {0};
                                    
    double correct_output_arr[24] = { //i = 0, j = 0
                                      79,121,
                                      184,291,

                                      //i = 0, j = 1
                                      277,319,361,403,
                                      692,799,906,1013,


									  //i = 1, j = 0
                                      289,461,
                                      394,631,

									  //i = 1, j = 1
                                      1107,1279,1451,1623, 
                                      1522,1759,1996,2233};

    //Bispace for i 
    sparse_bispace<1> spb_i(4);
    std::vector<size_t> split_points_i;
    split_points_i.push_back(2);
    spb_i.split(split_points_i);

    //Bispace for k 
    sparse_bispace<1> spb_k(5);
    std::vector<size_t> split_points_k;
    split_points_k.push_back(2);
    spb_k.split(split_points_k);
    
    //Bispace for j
    sparse_bispace<1> spb_j(6);
    std::vector<size_t> split_points_j;
    split_points_j.push_back(2);
    spb_j.split(split_points_j);

    sparse_bispace<2> A_spb = spb_i | spb_k;
    sparse_bispace<2> B_spb = spb_j | spb_k;
    sparse_bispace<2> C_spb = spb_i | spb_j;


    //For block_contract2_kernel
    std::vector< sequence<1,size_t> > output_indices_sets;
    std::vector< sequence<2,size_t> > input_indices_sets;
    std::vector< sequence<1,bool> > output_ignore_sets;
    std::vector< sequence<2,bool> > input_ignore_sets;


    std::vector< block_loop<1,2> > loop_list; 

    //i loop
    sequence<1,size_t> i_output_bispace_indices(0);
    sequence<2,size_t> i_input_bispace_indices(0); //B ignored
    sequence<1,bool> i_output_ignore(false);
    sequence<2,bool> i_input_ignore(false);
    i_input_ignore[1] = true;
    loop_list.push_back(block_loop<1,2>(i_output_bispace_indices,
                                        i_input_bispace_indices,
                                        i_output_ignore,
                                        i_input_ignore));
    output_indices_sets.push_back(i_output_bispace_indices);
    input_indices_sets.push_back(i_input_bispace_indices);
    output_ignore_sets.push_back(i_output_ignore);
    input_ignore_sets.push_back(i_input_ignore);


    //j loop
    sequence<1,size_t> j_output_bispace_indices(1);
    sequence<2,size_t> j_input_bispace_indices(0); //A ignored
    sequence<1,bool> j_output_ignore(false);
    sequence<2,bool> j_input_ignore(true);
    j_input_ignore[1] = false;
    loop_list.push_back(block_loop<1,2>(j_output_bispace_indices,
                                        j_input_bispace_indices,
                                        j_output_ignore,
                                        j_input_ignore));
    output_indices_sets.push_back(j_output_bispace_indices);
    input_indices_sets.push_back(j_input_bispace_indices);
    output_ignore_sets.push_back(j_output_ignore);
    input_ignore_sets.push_back(j_input_ignore);

    //k loop
    sequence<1,size_t> k_output_bispace_indices; //C ignored
    sequence<2,size_t> k_input_bispace_indices(1);
    sequence<1,bool> k_output_ignore(true);
    sequence<2,bool> k_input_ignore(false);
    loop_list.push_back(block_loop<1,2>(k_output_bispace_indices,
                                        k_input_bispace_indices,
                                        k_output_ignore,
                                        k_input_ignore));
    output_indices_sets.push_back(k_output_bispace_indices);
    input_indices_sets.push_back(k_input_bispace_indices);
    output_ignore_sets.push_back(k_output_ignore);
    input_ignore_sets.push_back(k_input_ignore);

    block_contract2_kernel<double> bc2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets);

    sequence<1,double*> output_ptrs(test_output_arr); 
    sequence<2,const double*> input_ptrs(test_input_arr_1); 
    input_ptrs[1] = test_input_arr_2;
    sequence<1,sparse_bispace_any_order> output_bispaces(C_spb);
    sequence<2,sparse_bispace_any_order> input_bispaces;
    input_bispaces[0] = A_spb;
    input_bispaces[1] = B_spb;

    run_loop_list(loop_list,bc2k,output_ptrs,input_ptrs,output_bispaces,input_bispaces);
    for(int i = 0; i < 24; ++i)
    {
        if(test_output_arr[i] != correct_output_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_loop<M,N,T>::run(...) produced incorrect output");
        }
    }
}


void block_loop_test::test_run_block_contract2_kernel_3d_2d() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_run_block_contract2_kernel_3d_2d()";

	//Indices in comments are block indices
    //dimensions: i = 3,j = 4, k = 5,l = 6
    //Contraction takes the form of A*B
    double test_input_arr_1[60] = {//i = 0 j = 0 k = 0 (1,2,2)
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

                                   //i = 1 j = 0 k = 0 (2,2,2)
                                   21,22,
                                   23,24,
                                   25,26,
                                   27,28,

                                   //i = 1 j = 0 k = 1 (2,2,3)
                                   29,30,31,
                                   32,33,34,
                                   35,36,37,
                                   38,39,40,

                                   //i = 1 j = 1 k = 0 (2,2,2)
                                   41,42,
                                   43,44,
                                   45,46,
                                   47,48,


                                   //i = 1 j = 1 k = 1 (2,2,3)
                                   49,50,51,
                                   52,53,54,
                                   55,56,57,
                                   58,59,60};


    double test_input_arr_2[30] = {//k = 0  l = 0
                                   1,2,3,
                                   4,5,6,

                                   //k = 0 l = 1
                                   7,8,9,
                                   10,11,12,

                                   //k = 1 l = 0
                                   13,14,15,
                                   16,17,18,
                                   19,20,21,

                                   //k = 1 l = 1
                                   22,23,24,
                                   25,26,27,
                                   28,29,30};

    double correct_output_arr[72] = {//i = 0 j = 0 l = 0 
                                     303,324,345,
                                     457,491,525,

                                     //i = 0 j = 0 l = 1
                                     483,504,525, 
                                     742,776,810,

                                     //i = 0 j = 1 l = 0 
                                     833,904,975,
                                     987,1071,1155,

                                     //i = 0 j = 1 l = 1
                                     1403,1474,1545,
                                     1662,1746,1830,

                                     //i = 1 j = 0 l = 0
                                     1555,1688,1821,
                                     1709,1855,2001,
                                     1863,2022,2181,
                                     2017,2189,2361,

                                     //i = 1 j = 0 l = 1
                                     2623,2756,2889,
                                     2882,3028,3174,
                                     3141,3300,3459,
                                     3400,3572,3744,

                                     //i = 1 j = 1 l = 0
                                     2615,2848,3081, 
                                     2769,3015,3261,
                                     2923,3182,3441,
                                     3077,3349,3621,

                                     //i = 1 j = 1 l = 1
                                     4463,4696,4929,
                                     4722,4968,5214, 
                                     4981,5240,5499,
                                     5240,5512,5784};


    double test_output_arr[72] = {0};

    //Bispace for i 
    sparse_bispace<1> spb_i(3);
    std::vector<size_t> split_points_i;
    split_points_i.push_back(1);
    spb_i.split(split_points_i);

    //Bispace for j 
    sparse_bispace<1> spb_j(4);
    std::vector<size_t> split_points_j;
    split_points_j.push_back(2);
    spb_j.split(split_points_j);

    //Bispace for k 
    sparse_bispace<1> spb_k(5);
    std::vector<size_t> split_points_k;
    split_points_k.push_back(2);
    spb_k.split(split_points_k);

    //Bispace for l 
    sparse_bispace<1> spb_l(6);
    std::vector<size_t> split_points_l;
    split_points_l.push_back(3);
    spb_l.split(split_points_l);

    sparse_bispace<3> A_spb = spb_i | spb_j | spb_k;
    sparse_bispace<2> B_spb = spb_k | spb_l;
    sparse_bispace<3> C_spb = spb_i | spb_j | spb_l;


    //For block_contract2_kernel
    std::vector< sequence<1,size_t> > output_indices_sets;
    std::vector< sequence<2,size_t> > input_indices_sets;
    std::vector< sequence<1,bool> > output_ignore_sets;
    std::vector< sequence<2,bool> > input_ignore_sets;


    std::vector< block_loop<1,2> > loop_list; 

    //i loop
    sequence<1,size_t> i_output_bispace_indices(0);
    sequence<2,size_t> i_input_bispace_indices(0); //B ignored
    sequence<1,bool> i_output_ignore(false);
    sequence<2,bool> i_input_ignore(false);
    i_input_ignore[1] = true;
    loop_list.push_back(block_loop<1,2>(i_output_bispace_indices,
                                        i_input_bispace_indices,
                                        i_output_ignore,
                                        i_input_ignore));
    output_indices_sets.push_back(i_output_bispace_indices);
    input_indices_sets.push_back(i_input_bispace_indices);
    output_ignore_sets.push_back(i_output_ignore);
    input_ignore_sets.push_back(i_input_ignore);


    //j loop
    sequence<1,size_t> j_output_bispace_indices(1);
    sequence<2,size_t> j_input_bispace_indices(1); //B ignored
    sequence<1,bool> j_output_ignore(false);
    sequence<2,bool> j_input_ignore(false);
    j_input_ignore[1] = true;
    loop_list.push_back(block_loop<1,2>(j_output_bispace_indices,
                                        j_input_bispace_indices,
                                        j_output_ignore,
                                        j_input_ignore));
    output_indices_sets.push_back(j_output_bispace_indices);
    input_indices_sets.push_back(j_input_bispace_indices);
    output_ignore_sets.push_back(j_output_ignore);
    input_ignore_sets.push_back(j_input_ignore);

    //l loop
    sequence<1,size_t> l_output_bispace_indices(2);
    sequence<2,size_t> l_input_bispace_indices(1); //A ignored
    sequence<1,bool> l_output_ignore(false);
    sequence<2,bool> l_input_ignore(true);
    l_input_ignore[1] = false;
    loop_list.push_back(block_loop<1,2>(l_output_bispace_indices,
                                        l_input_bispace_indices,
                                        l_output_ignore,
                                        l_input_ignore));
    output_indices_sets.push_back(l_output_bispace_indices);
    input_indices_sets.push_back(l_input_bispace_indices);
    output_ignore_sets.push_back(l_output_ignore);
    input_ignore_sets.push_back(l_input_ignore);

    //k loop
    sequence<1,size_t> k_output_bispace_indices; //C ignored
    sequence<2,size_t> k_input_bispace_indices(2);
    k_input_bispace_indices[1] = 0;
    sequence<1,bool> k_output_ignore(true);
    sequence<2,bool> k_input_ignore(false);
    loop_list.push_back(block_loop<1,2>(k_output_bispace_indices,
                                        k_input_bispace_indices,
                                        k_output_ignore,
                                        k_input_ignore));
    output_indices_sets.push_back(k_output_bispace_indices);
    input_indices_sets.push_back(k_input_bispace_indices);
    output_ignore_sets.push_back(k_output_ignore);
    input_ignore_sets.push_back(k_input_ignore);

    block_contract2_kernel<double> bc2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets);

    sequence<1,double*> output_ptrs(test_output_arr); 
    sequence<2,const double*> input_ptrs(test_input_arr_1); 
    input_ptrs[1] = test_input_arr_2;
    sequence<1,sparse_bispace_any_order> output_bispaces(C_spb);
    sequence<2,sparse_bispace_any_order> input_bispaces;
    input_bispaces[0] = A_spb;
    input_bispaces[1] = B_spb;

    run_loop_list(loop_list,bc2k,output_ptrs,input_ptrs,output_bispaces,input_bispaces);

    for(int i = 0; i < 72; ++i)
    {
        if(test_output_arr[i] != correct_output_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_loop<M,N,T>::run(...) produced incorrect output");
        }
    }
}

} // namespace libtensor
