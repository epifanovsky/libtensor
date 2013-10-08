#include <libtensor/block_sparse/block_loop.h>
#include <libtensor/block_sparse/block_kernels.h>
#include <sstream>
#include "block_loop_test.h" 

//TODO REMOVE
#include <iostream> 


namespace libtensor {

void block_loop_test::perform() throw(libtest::test_exception) {

    test_range();

    test_run_block_copy_kernel_1d();
    test_run_block_copy_kernel_2d();
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
    block_loop<1,1> bl(spb,
                       output_bispace_indices,
                       input_bispace_indices,
                       output_ignore,
                       input_ignore,
                       bck);


    double test_output_arr[8];
    double test_input_arr[8] = {0,1,2,3,5,6,7};

    sequence<1,double*> output_ptrs(test_output_arr); 
    sequence<1,double*> input_ptrs(test_input_arr); 


    sequence<1, sparse_bispace_generic_i*> output_bispaces(&spb);
    sequence<1, sparse_bispace_generic_i*> input_bispaces(&spb);

    bl.run(output_ptrs,input_ptrs,output_bispaces,input_bispaces);

    for(int i = 0; i < 8; ++i)
    {
        if(test_output_arr[i] != test_input_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_loop<M,N,T>::run(...) produced incorrect output");
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

    block_loop<1,1> bl_1(spb_1,
                       output_bispace_indices_1,
                       input_bispace_indices_1,
                       output_ignore_1,
                       input_ignore_1,
                       bck);

    bl_1.nest(spb_2,
              output_bispace_indices_2,
              input_bispace_indices_2,
              output_ignore_2,
              input_ignore_2);

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

    bl_1.run(output_ptrs,input_ptrs,output_bispaces,input_bispaces);

    for(int i = 0; i < 20; ++i)
    {
        if(test_output_arr[i] != test_input_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_loop<M,N,T>::run(...) produced incorrect output");
        }
    }

}

#if 0
//Copy and transpose a 2d array
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

    block_loop<1,1> bl_1(spb_1,
                       output_bispace_indices_1,
                       input_bispace_indices_1,
                       output_ignore_1,
                       input_ignore_1,
                       bck);

    bl_1.nest(spb_2,
              output_bispace_indices_2,
              input_bispace_indices_2,
              output_ignore_2,
              input_ignore_2);

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

    bl_1.run(output_ptrs,input_ptrs,output_bispaces,input_bispaces);

    for(int i = 0; i < 20; ++i)
    {
        if(test_output_arr[i] != test_input_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_loop<M,N,T>::run(...) produced incorrect output");
        }
    }

}
#endif

} // namespace libtensor
