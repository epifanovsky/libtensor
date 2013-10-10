#include <libtensor/block_sparse/block_kernels.h>
#include <sstream>
#include "block_kernels_test.h" 

//TODO REMOVE
#include <iostream>

namespace libtensor {

void block_kernels_test::perform() throw(libtest::test_exception) {

    test_block_printer_2d();
    test_block_printer_3d();

    test_block_copy_kernel_2d();
    test_block_copy_kernel_3d();

    test_block_equality_kernel_2d_true();
    test_block_equality_kernel_2d_false();
    test_block_equality_kernel_not_run_exception();


    test_block_permute_kernel_2d();
    test_block_permute_kernel_2d_invalid_perm_incomplete();
    test_block_permute_kernel_2d_invalid_perm_oob();
    test_block_permute_kernel_2d_invalid_perm_duplicate();
    test_block_permute_kernel_3d_201();
    test_block_permute_kernel_3d_021();
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
    std::string correct_str("---\n 1 2\n 3 4\n");

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
    std::string correct_str("---\n 1 2\n 3 4\n\n 5 6\n 7 8\n");

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

void block_kernels_test::test_block_permute_kernel_2d() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d()";

    double test_input_block[4] = {1,2,
                                  3,4};

    double correct_output_block[4] = {1,3,
                                      2,4};
    double test_output_block[4];

    permute_map perm;
    perm.insert(std::make_pair(1,0)); 
    perm.insert(std::make_pair(0,1)); 

    block_permute_kernel<double> b_perm_k(perm);


    dim_list dims;
    dims.push_back(2);
    dims.push_back(2);

    sequence<1,double*> output_ptrs(test_output_block);
    sequence<1,double*> input_ptrs(test_input_block);
    sequence<1,dim_list> output_dims(dims);
    sequence<1,dim_list> input_dims(dims);


    b_perm_k(output_ptrs,input_ptrs,output_dims,input_dims);
    for(int i = 0; i < 4; ++i)
    {
        if(test_output_block[i] != correct_output_block[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_permute_kernel::operator(...) did not produce correct result");
        }
    }
}

//Should throw an exception if we try to construct a kernel with an incomplete permutation 
void block_kernels_test::test_block_permute_kernel_2d_invalid_perm_incomplete() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_invalid_perm_incomplete()";

    permute_map perm;
    perm.insert(std::make_pair(1,0)); 

    bool threw_exception = false;
    try
    {
        block_permute_kernel<double> b_perm_k(perm);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_permute_kernel<T>::block_permute_kernel(...) did not throw exception when incomplete permutation map supplied");
    }
}

//Should throw an exception if we try to process a block with a permutation that lies out of bounds
void block_kernels_test::test_block_permute_kernel_2d_invalid_perm_oob() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_invalid_perm_oob()";

    double test_input_block[4] = {1,2,
                                  3,4};
    double test_output_block[4];

    permute_map perm;
    perm.insert(std::make_pair(2,0)); 
    perm.insert(std::make_pair(0,2)); 

    block_permute_kernel<double> b_perm_k(perm);


    dim_list dims;
    dims.push_back(2);
    dims.push_back(2);

    sequence<1,double*> output_ptrs(test_output_block);
    sequence<1,double*> input_ptrs(test_input_block);
    sequence<1,dim_list> output_dims;
    sequence<1,dim_list> input_dims(dims);

    bool threw_exception = false;
    try
    {
        b_perm_k(output_ptrs,input_ptrs,output_dims,input_dims);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_permute_kernel::operator()(...) did not throw exception when out of bounds permutation map supplied");
    }
}

//Should throw an exception if we pass a permutation that maps two indices to the same location
void block_kernels_test::test_block_permute_kernel_2d_invalid_perm_duplicate() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_invalid_perm_duplicate()";

    permute_map perm;
    perm.insert(std::make_pair(2,1)); 
    perm.insert(std::make_pair(0,1)); 
    perm.insert(std::make_pair(1,0)); 

    bool threw_exception = false;
    try
    {
        block_permute_kernel<double> b_perm_k(perm);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_permute_kernel<T>::block_permute_kernel(...) did not throw exception when permutation map with duplicate indices supplied");
    }
}

//A more complicated 3D permutation, with indices permuted in cyclic order
void block_kernels_test::test_block_permute_kernel_3d_201() throw(libtest::test_exception)
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

    permute_map perm;
    perm.insert(std::make_pair(1,0)); 
    perm.insert(std::make_pair(2,1)); 
    perm.insert(std::make_pair(0,2)); 

    block_permute_kernel<double> b_perm_k(perm);

    dim_list dims;
    dims.push_back(4);
    dims.push_back(2);
    dims.push_back(3);

    sequence<1,double*> output_ptrs(test_output_block);
    sequence<1,double*> input_ptrs(test_input_block);
    sequence<1,dim_list> output_dims(dims);
    sequence<1,dim_list> input_dims(dims);


    b_perm_k(output_ptrs,input_ptrs,output_dims,input_dims);
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

    permute_map perm;
    perm.insert(std::make_pair(2,1)); 
    perm.insert(std::make_pair(1,2)); 

    block_permute_kernel<double> b_perm_k(perm);

    dim_list dims;
    dims.push_back(4);
    dims.push_back(2);
    dims.push_back(3);

    sequence<1,double*> output_ptrs(test_output_block);
    sequence<1,double*> input_ptrs(test_input_block);
    sequence<1,dim_list> output_dims(dims);
    sequence<1,dim_list> input_dims(dims);


    b_perm_k(output_ptrs,input_ptrs,output_dims,input_dims);
    for(int i = 0; i < 24; ++i)
    {
        if(test_output_block[i] != correct_output_block[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_permute_kernel::operator(...) did not produce correct result");
        }
    }
}

} // namespace libtensor
