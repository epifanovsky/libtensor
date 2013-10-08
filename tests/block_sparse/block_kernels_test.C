#include <libtensor/block_sparse/block_kernels.h>
#include <sstream>
#include "block_kernels_test.h" 

namespace libtensor {

void block_kernels_test::perform() throw(libtest::test_exception) {

    test_block_printer_2d();
    test_block_printer_3d();

    test_block_copy_kernel_2d();
    test_block_copy_kernel_3d();

    test_block_equality_kernel_2d_true();
    test_block_equality_kernel_2d_false();
    test_block_equality_kernel_not_run_exception();
}


void block_kernels_test::test_block_printer_2d() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_printer_2d()";
    block_printer<double> bp;

    double test_block_arr[4] = {1,2,
                                3,4};
    dim_list dims;
    dims.push_back(2); 
    dims.push_back(2); 

    sequence<0, dim_list > output_dims;  
    sequence<1, dim_list > input_dims(dims);
    sequence<0, double*> output_ptrs;
    sequence<1, double*> input_ptrs(test_block_arr);

    bp(output_ptrs,input_ptrs,output_dims,input_dims);
    std::string correct_str(" 1 2\n 3 4\n");

    if(!(bp.str()  == correct_str))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_printer::str(...) returned incorrect value");
    }
}


void block_kernels_test::test_block_printer_3d() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_printer_3d()";
    //TODO remove superflous double specifiers replace with arguments
    block_printer<double> bp;

    double test_block_arr[8] = {1,2,
                                3,4,
 
                                5,6,
                                7,8};
    dim_list dims;
    dims.push_back(2); 
    dims.push_back(2); 
    dims.push_back(2); 
    sequence<0, dim_list> output_dims;
    sequence<1, dim_list> input_dims(dims);
    sequence<0, double*> output_ptrs;
    sequence<1, double*> input_ptrs(test_block_arr);


    bp(output_ptrs,input_ptrs,output_dims,input_dims);
    std::string correct_str(" 1 2\n 3 4\n\n 5 6\n 7 8\n");

    if(!(bp.str()  == correct_str))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_printer::str(...) returned incorrect value");
    }
}

void block_kernels_test::test_block_copy_kernel_2d() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_copy_kernel_2d()";
    block_copy_kernel<double> bck;

    double test_input_block[4] = {1,2,
                                  3,4};
    double test_output_block[4];

    std::vector<size_t> dims;
    dims.push_back(2);
    dims.push_back(2);

    sequence<1,double*> output_ptrs(test_output_block);
    sequence<1,double*> input_ptrs(test_input_block);
    sequence<1,dim_list> output_dims(dims);
    sequence<1,dim_list> input_dims(dims);

    bck(output_ptrs,input_ptrs,output_dims,input_dims);
    
    for(int i = 0; i < 4; ++i)
    {
        if(test_output_block[i] != test_input_block[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_copy_kernel::operator(...) did not produce correct result");
        }
    }
}

void block_kernels_test::test_block_copy_kernel_3d() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_copy_kernel_3d()";
    block_copy_kernel<double> bck;

    double test_input_block[12] = {1,2,
                                   3,4,
    
                                   5,6,
                                   7,8, 
                                  
                                   9,10,
                                  11,12};
    double test_output_block[12];

    dim_list dims; 
    dims.push_back(3);
    dims.push_back(2);
    dims.push_back(2);
    
    sequence<1,double*> output_ptrs(test_output_block);
    sequence<1,double*> input_ptrs(test_input_block);
    sequence<1,dim_list> output_dims(dims);
    sequence<1,dim_list> input_dims(dims);


    bck(output_ptrs,input_ptrs,output_dims,input_dims);
    
    for(int i = 0; i < 12; ++i)
    {
        if(test_output_block[i] != test_input_block[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_copy_kernel::operator(...) did not produce correct result");
        }
    }
}

void block_kernels_test::test_block_equality_kernel_2d_true() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_equality_kernel_2d()_true";
    block_equality_kernel<double> bek;

    double test_input_block_1[4] = {1,2,
                                    3,4};
    double test_input_block_2[4] = {1,2,
                                    3,4};

    std::vector<size_t> dims;
    dims.push_back(2);
    dims.push_back(2);

    sequence<0,double*> output_ptrs;
    sequence<2,double*> input_ptrs(test_input_block_1);
    input_ptrs[1] = test_input_block_2;

    sequence<0,dim_list> output_dims;
    sequence<2,dim_list> input_dims(dims);

    bek(output_ptrs,input_ptrs,output_dims,input_dims);
    
    if(!bek.equal())
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_equality_kernel::operator(...) did not produce correct result");

    }
}

void block_kernels_test::test_block_equality_kernel_2d_false() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_equality_kernel_2d_false()";
    block_equality_kernel<double> bek;

    double test_input_block_1[4] = {1,2,
                                    3,4};

    //Note the '5' instead of '4'
    double test_input_block_2[4] = {1,2,
                                    3,5};
    std::vector<size_t> dims;
    dims.push_back(2);
    dims.push_back(2);

    sequence<0,double*> output_ptrs;
    sequence<2,double*> input_ptrs(test_input_block_1);
    input_ptrs[1] = test_input_block_2;

    sequence<0,dim_list> output_dims;
    sequence<2,dim_list> input_dims(dims);

    bek(output_ptrs,input_ptrs,output_dims,input_dims);

    if(bek.equal())
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_equality_kernel::operator(...) did not produce correct result");

    }
}

void block_kernels_test::test_block_equality_kernel_not_run_exception() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_equality_kernel_not_run_exception()";
    bool threw_exception = false;
    block_equality_kernel<double> bek;

    double test_input_block_1[4] = {1,2,
                                    3,4};

    //Note the '5' instead of '4'
    double test_input_block_2[4] = {1,2,
                                    3,5};


    try
    {
        bek.equal();
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_equality_kernel::equal(...) did not throw exception when no values compared");
    }
}

} // namespace libtensor
