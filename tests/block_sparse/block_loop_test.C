#include <libtensor/block_sparse/block_loop.h>
#include <libtensor/block_sparse/block_kernels.h>
#include <sstream>
#include "block_loop_test.h" 

//TODO REMOVE
#include <iostream> 


namespace libtensor {

void block_loop_test::perform() throw(libtest::test_exception) {

    test_range();

    test_run_invalid_bispaces();

    test_run_block_copy_kernel_1d();
    test_run_block_copy_kernel_2d();

    test_run_block_permute_kernel_2d();
    test_run_block_permute_kernel_3d_201();
}

void block_loop_test::test_range() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_range()";
    block_list the_range = range(0,5);
    block_list correct_range;
    correct_range.push_back(0);
    correct_range.push_back(1);
    correct_range.push_back(2);
    correct_range.push_back(3);
    correct_range.push_back(4);

    if(the_range.size() != correct_range.size())
    {
        fail_test(test_name,__FILE__,__LINE__,
                "range(...) returned incorrect value");
    }

    for(int i = 0; i < 5; ++i)
    {
        if(the_range[i] != correct_range[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "range(...) returned incorrect value");
        }
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

	permute_map perm;
	perm.insert(std::make_pair(0,1));
	perm.insert(std::make_pair(1,0));

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
    sequence<1,double*> input_ptrs(test_input_arr); 

    //Here, one of the bispaces should be transposed but is not... this should cause an exception because the bispaces
    //do not line up in the loops
    sequence<1,sparse_bispace_generic_i*> output_bispaces(&two_d);
    sequence<1,sparse_bispace_generic_i*> input_bispaces(&two_d);

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

void block_loop_test::test_run_block_copy_kernel_1d() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_run_block_copy_kernel_1d()";

    sparse_bispace<1> spb(8);
    std::vector<size_t> split_points;
    split_points.push_back(2);
    split_points.push_back(5);
    spb.split(split_points);
    std::vector<sparse_bispace<1> > spb_vec(1,spb);

    sequence<1,size_t> output_bispace_indices(0);
    sequence<1,size_t> input_bispace_indices(0);

    sequence<1,bool> output_ignore(false); 
    sequence<1,bool> input_ignore(false); 

    block_copy_kernel<double> bck;
    block_loop<1,1> bl(output_bispace_indices,
                       input_bispace_indices,
                       output_ignore,
                       input_ignore);


    double test_output_arr[8];
    double test_input_arr[8] = {0,1,2,3,5,6,7};

    sequence<1,double*> output_ptrs(test_output_arr); 
    sequence<1,double*> input_ptrs(test_input_arr); 


    sequence<1, sparse_bispace_generic_i*> output_bispaces(&spb);
    sequence<1, sparse_bispace_generic_i*> input_bispaces(&spb);

    run_loop_list(bl,bck,output_ptrs,input_ptrs,output_bispaces,input_bispaces);

    for(int i = 0; i < 8; ++i)
    {
        if(test_output_arr[i] != test_input_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "run_loop_list(...) produced incorrect output");
        }
    }
}

//First nested loop test
//Copy a 2d block-major array
void block_loop_test::test_run_block_copy_kernel_2d() throw(libtest::test_exception)
{
    static const char *test_name = "block_loop_test::test_run_block_copy_kernel_2d()";

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

    sparse_bispace<2> two_d = spb_1 | spb_2;

    block_copy_kernel<double> bck;

    sequence<1,size_t> output_bispace_indices_1(0);
    sequence<1,size_t> input_bispace_indices_1(0);
    sequence<1,bool> output_ignore_1(false);
    sequence<1,bool> input_ignore_1(false);

    sequence<1,size_t> output_bispace_indices_2(1);
    sequence<1,size_t> input_bispace_indices_2(1);
    sequence<1,bool> output_ignore_2(false);
    sequence<1,bool> input_ignore_2(false);

    std::vector< block_loop<1,1> > loop_list;
    loop_list.push_back(block_loop<1,1>(output_bispace_indices_1,
						input_bispace_indices_1,
						output_ignore_1,
						input_ignore_1));

    loop_list.push_back(block_loop<1,1>(output_bispace_indices_2,
                        input_bispace_indices_2,
                        output_ignore_2,
                        input_ignore_2));


    double test_output_arr[20];
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
                                 18,19,20
                                };

    sequence<1,double*> output_ptrs(test_output_arr); 
    sequence<1,double*> input_ptrs(test_input_arr); 
    sequence<1,sparse_bispace_generic_i*> output_bispaces(&two_d);
    sequence<1,sparse_bispace_generic_i*> input_bispaces(&two_d);

    run_loop_list(loop_list,bck,output_ptrs,input_ptrs,output_bispaces,input_bispaces);

    for(int i = 0; i < 20; ++i)
    {
        if(test_output_arr[i] != test_input_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "run_loop_list(...) produced incorrect output");
        }
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

	permute_map perm;
	perm.insert(std::make_pair(0,1));
	perm.insert(std::make_pair(1,0));

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
    sequence<1,double*> input_ptrs(test_input_arr); 
    sequence<1,sparse_bispace_generic_i*> output_bispaces(&two_d_output);
    sequence<1,sparse_bispace_generic_i*> input_bispaces(&two_d_input);

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

	permute_map perm;
	perm.insert(std::make_pair(0,2));
	perm.insert(std::make_pair(1,0));
	perm.insert(std::make_pair(2,1));

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
    sequence<1,double*> input_ptrs(test_input_arr); 
    sequence<1,sparse_bispace_generic_i*> output_bispaces(&three_d_output);
    sequence<1,sparse_bispace_generic_i*> input_bispaces(&three_d_input);

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

} // namespace libtensor
