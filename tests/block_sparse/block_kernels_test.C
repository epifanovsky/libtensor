#include <libtensor/block_sparse/block_kernels.h>
#include <libtensor/block_sparse/sparse_bispace.h>
#include <libtensor/iface/letter.h>
#include <sstream>
#include "block_kernels_test.h" 

//TODO REMOVE
#include <iostream>

namespace libtensor {

void block_kernels_test::perform() throw(libtest::test_exception) {

    test_block_print_kernel_2d();
    test_block_print_kernel_3d();

    test_block_permute_kernel_2d();
    test_block_permute_kernel_3d_201();
    test_block_permute_kernel_3d_021();

    test_block_contract2_kernel_2d_not_enough_indices();
    test_block_contract2_kernel_2d_strided_output();
    test_block_contract2_kernel_2d_oob_indices();
    test_block_contract2_kernel_2d_not_matching_indices();
    test_block_contract2_kernel_2d_wrong_dim_order();
    test_block_contract2_kernel_2d_ip_pj();
    test_block_contract2_kernel_2d_ip_jp();
    test_block_contract2_kernel_2d_pi_pj();
    test_block_contract2_kernel_2d_pi_jp();
    test_block_contract2_kernel_3d_2d();
    test_block_contract2_kernel_3d_3d_multi_index();

    test_block_subtract_kernel_2d_2d();

    /*test_direct_block_subtract_kernel_2d_2d();*/
}


void block_kernels_test::test_block_print_kernel_2d() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_print_kernel_2d()";

    double test_block_arr[4] = {1,2,
                                3,4};
    std::vector<double*> ptrs(1,test_block_arr);

    dim_list dims;
    dims.push_back(2); 
    dims.push_back(2); 
    std::vector<dim_list> dim_lists(1,dims);

    block_print_kernel<double> bpk;
    bpk(ptrs,dim_lists);
    std::string correct_str("---\n 1 2\n 3 4\n");

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
    std::vector<double*> ptrs(1,test_block_arr);

    dim_list dims;
    dims.push_back(2); 
    dims.push_back(2); 
    dims.push_back(2); 
    std::vector<dim_list> dim_lists(1,dims);

    block_print_kernel<double> bpk;
    bpk(ptrs,dim_lists);
    std::string correct_str("---\n 1 2\n 3 4\n\n 5 6\n 7 8\n");
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

    runtime_permutation perm(2);
    perm.permute(0,1);
    block_permute_kernel<double> b_perm_k(perm);


    dim_list dims;
    dims.push_back(2);
    dims.push_back(2);

    sequence<1,double*> output_ptrs(test_output_block);
    sequence<1,const double*> input_ptrs(test_input_block);
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

    runtime_permutation perm(3);
    perm.permute(0,2);
    perm.permute(0,1);

    block_permute_kernel<double> b_perm_k(perm);

    dim_list dims;
    dims.push_back(4);
    dims.push_back(2);
    dims.push_back(3);

    sequence<1,double*> output_ptrs(test_output_block);
    sequence<1,const double*> input_ptrs(test_input_block);
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

    runtime_permutation perm(3);
    perm.permute(1,2);
    block_permute_kernel<double> b_perm_k(perm);

    dim_list dims;
    dims.push_back(4);
    dims.push_back(2);
    dims.push_back(3);

    sequence<1,double*> output_ptrs(test_output_block);
    sequence<1,const double*> input_ptrs(test_input_block);
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

//Should throw an exception because there is an entry missing from the input indices
void block_kernels_test::test_block_contract2_kernel_2d_not_enough_indices() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_not_enough_indices()";

    //C_ij = \sum_k A_ik B_kj
    //dimensions: i = 2,j = 3,k = 4
    //Just need dummy data for this test
    double test_output_block[6]; //C
    double test_input_block_1[8]; //A 
    double test_input_block_2[12]; //B
    
    //Initialize the indices touched by each loop
    std::vector< sequence<1,size_t> > output_indices_sets;
    std::vector< sequence<2,size_t> > input_indices_sets;
    std::vector< sequence<1,bool> > output_ignore_sets;
    std::vector< sequence<2,bool> > input_ignore_sets;

    //i loop
    sequence<1,size_t> i_output_indices(0);
    sequence<2,size_t> i_input_indices(0); //B is ignored

    sequence<1,bool> i_output_ignore(false);
    sequence<2,bool> i_input_ignore(false);
    i_input_ignore[1] = true;

    //j loop
    sequence<1,size_t> j_output_indices(1);
    sequence<2,size_t> j_input_indices(1); //A is ignored

    sequence<1,bool> j_output_ignore(false); 
    sequence<2,bool> j_input_ignore(false);
    j_input_ignore[0] = true;

    //k loop
    sequence<1,size_t> k_output_indices;
    sequence<2,size_t> k_input_indices(1);
    k_input_indices[1] = 0;

    sequence<1,bool> k_output_ignore(true);
    sequence<2,bool> k_input_ignore(false);


    output_indices_sets.push_back(i_output_indices);
    output_indices_sets.push_back(j_output_indices);
    output_indices_sets.push_back(k_output_indices);
    input_indices_sets.push_back(i_input_indices);
    input_indices_sets.push_back(j_input_indices);
    //Deliberately forget to add the k index to the ignore list

    output_ignore_sets.push_back(i_output_ignore);
    output_ignore_sets.push_back(j_output_ignore);
    output_ignore_sets.push_back(k_output_ignore);
    input_ignore_sets.push_back(i_input_ignore);
    input_ignore_sets.push_back(j_input_ignore);
    input_ignore_sets.push_back(k_input_ignore);

    bool threw_exception = false;
    try
    {
        block_contract2_kernel<double> bc2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets); 
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::block_contract2_kernel(...) did not throw exception when indices_sets element omitted");
    }
}

//We don't allow output to be strided in the inner kernel - just dumb...
void block_kernels_test::test_block_contract2_kernel_2d_strided_output() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_strided_output()";

    //Output strided test
    //C_ji = \sum_k A_ik B_kj
    //dimensions: i = 2,j = 3,k = 4
    //Just need dummy data for this test
    double test_output_block[6]; //C
    double test_input_block_1[8]; //A 
    double test_input_block_2[12]; //B
    
    //Initialize the indices touched by each loop
    std::vector< sequence<1,size_t> > output_indices_sets;
    std::vector< sequence<2,size_t> > input_indices_sets;
    std::vector< sequence<1,bool> > output_ignore_sets;
    std::vector< sequence<2,bool> > input_ignore_sets;

    //i loop
    sequence<1,size_t> i_output_indices(1); //We stride here
    sequence<2,size_t> i_input_indices(0); //B is ignored

    sequence<1,bool> i_output_ignore(false);
    sequence<2,bool> i_input_ignore(false);
    i_input_ignore[1] = true;

    //j loop
    sequence<1,size_t> j_output_indices(0); //and stride here...
    sequence<2,size_t> j_input_indices(1); //A is ignored

    sequence<1,bool> j_output_ignore(false); 
    sequence<2,bool> j_input_ignore(false);
    j_input_ignore[0] = true;

    //k loop
    sequence<1,size_t> k_output_indices;
    sequence<2,size_t> k_input_indices(1);
    k_input_indices[1] = 0;

    sequence<1,bool> k_output_ignore(true);
    sequence<2,bool> k_input_ignore(false);


    output_indices_sets.push_back(i_output_indices);
    output_indices_sets.push_back(j_output_indices);
    output_indices_sets.push_back(k_output_indices);
    input_indices_sets.push_back(i_input_indices);
    input_indices_sets.push_back(j_input_indices);
    input_indices_sets.push_back(k_input_indices);

    output_ignore_sets.push_back(i_output_ignore);
    output_ignore_sets.push_back(j_output_ignore);
    output_ignore_sets.push_back(k_output_ignore);
    input_ignore_sets.push_back(i_input_ignore);
    input_ignore_sets.push_back(j_input_ignore);
    input_ignore_sets.push_back(k_input_ignore);

    block_contract2_kernel<double> bc2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets); 

    dim_list output_dims_1; //C
    output_dims_1.push_back(3);
    output_dims_1.push_back(2);

    dim_list input_dims_1; //A
    input_dims_1.push_back(2);
    input_dims_1.push_back(4);

    dim_list input_dims_2; //B
    input_dims_2.push_back(4);
    input_dims_2.push_back(3);


    sequence<1,double*> output_ptrs(test_output_block);
    sequence<2,const double*> input_ptrs(test_input_block_1);
    input_ptrs[1] = test_input_block_2;

    sequence<1, dim_list> output_dims(output_dims_1);
    sequence<2, dim_list> input_dims;
    input_dims[0] = input_dims_1;
    input_dims[1] = input_dims_2;

    bool threw_exception = false;
    try
    {
        bc2k(output_ptrs,input_ptrs,output_dims,input_dims);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel<T>::operator()(...) did not throw exception when output is strided");
    }
}

//Should throw an exception when specifying a tensor index that is greater than the tensor dimensions
void block_kernels_test::test_block_contract2_kernel_2d_oob_indices() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_oob_indices()";

    //C_ij = \sum_k A_ik B_kj
    //dimensions: i = 2,j = 3,k = 4
    //Just need dummy data for this test
    double test_output_block[6]; //C
    double test_input_block_1[8]; //A 
    double test_input_block_2[12]; //B
    
    //Initialize the indices touched by each loop
    std::vector< sequence<1,size_t> > output_indices_sets;
    std::vector< sequence<2,size_t> > input_indices_sets;
    std::vector< sequence<1,bool> > output_ignore_sets;
    std::vector< sequence<2,bool> > input_ignore_sets;

    //i loop
    //!!! Mess up the output value, make it out of bounds for all tensors
    sequence<1,size_t> i_output_indices(2);
    sequence<2,size_t> i_input_indices(0); //B is ignored

    sequence<1,bool> i_output_ignore(false);
    sequence<2,bool> i_input_ignore(false);
    i_input_ignore[1] = true;

    //j loop
    sequence<1,size_t> j_output_indices(1);
    sequence<2,size_t> j_input_indices(1); //A is ignored

    sequence<1,bool> j_output_ignore(false); 
    sequence<2,bool> j_input_ignore(false);
    j_input_ignore[0] = true;

    //k loop
    sequence<1,size_t> k_output_indices;
    sequence<2,size_t> k_input_indices(1);
    k_input_indices[1] = 0;

    sequence<1,bool> k_output_ignore(true);
    sequence<2,bool> k_input_ignore(false);


    output_indices_sets.push_back(i_output_indices);
    output_indices_sets.push_back(j_output_indices);
    output_indices_sets.push_back(k_output_indices);
    input_indices_sets.push_back(i_input_indices);
    input_indices_sets.push_back(j_input_indices);
    input_indices_sets.push_back(k_input_indices);

    output_ignore_sets.push_back(i_output_ignore);
    output_ignore_sets.push_back(j_output_ignore);
    output_ignore_sets.push_back(k_output_ignore);
    input_ignore_sets.push_back(i_input_ignore);
    input_ignore_sets.push_back(j_input_ignore);
    input_ignore_sets.push_back(k_input_ignore);

    block_contract2_kernel<double> bc2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets); 


    dim_list output_dims_1; //C
    output_dims_1.push_back(2);
    output_dims_1.push_back(3);

    dim_list input_dims_1; //A
    input_dims_1.push_back(2);
    input_dims_1.push_back(4);

    dim_list input_dims_2; //B
    input_dims_2.push_back(4);
    input_dims_2.push_back(3);


    sequence<1,double*> output_ptrs(test_output_block);
    sequence<2,const double*> input_ptrs(test_input_block_1);
    input_ptrs[1] = test_input_block_2;

    sequence<1, dim_list> output_dims(output_dims_1);
    sequence<2, dim_list> input_dims;
    input_dims[0] = input_dims_1;
    input_dims[1] = input_dims_2;

    bool threw_exception = false;
    try
    {
        bc2k(output_ptrs,input_ptrs,output_dims,input_dims);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel::operator(...) did not throw exception when out of bounds indices given");
    }
}

//Should throw an exception because indices to not match up 
void block_kernels_test::test_block_contract2_kernel_2d_not_matching_indices() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_not_matching_indices()";

    //C_ij = \sum_k A_ik B_kj
    //dimensions: i = 2,j = 3,k = 4
    //Just need dummy data for this test
    double test_output_block[6]; //C
    double test_input_block_1[8]; //A 
    double test_input_block_2[12]; //B
    
    //Initialize the indices touched by each loop
    std::vector< sequence<1,size_t> > output_indices_sets;
    std::vector< sequence<2,size_t> > input_indices_sets;
    std::vector< sequence<1,bool> > output_ignore_sets;
    std::vector< sequence<2,bool> > input_ignore_sets;

    //i loop
    sequence<1,size_t> i_output_indices(0);
    sequence<2,size_t> i_input_indices(0); //B is ignored

    sequence<1,bool> i_output_ignore(false);
    sequence<2,bool> i_input_ignore(false);
    i_input_ignore[1] = true;

    //j loop
    sequence<1,size_t> j_output_indices(1);
    sequence<2,size_t> j_input_indices(1); //A is ignored

    sequence<1,bool> j_output_ignore(false); 
    sequence<2,bool> j_input_ignore(false);
    j_input_ignore[0] = true;

    //k loop
    sequence<1,size_t> k_output_indices;
    sequence<2,size_t> k_input_indices(1);
    k_input_indices[1] = 0;

    sequence<1,bool> k_output_ignore(true);
    sequence<2,bool> k_input_ignore(false);


    output_indices_sets.push_back(i_output_indices);
    output_indices_sets.push_back(j_output_indices);
    output_indices_sets.push_back(k_output_indices);
    input_indices_sets.push_back(i_input_indices);
    input_indices_sets.push_back(j_input_indices);
    input_indices_sets.push_back(k_input_indices);

    output_ignore_sets.push_back(i_output_ignore);
    output_ignore_sets.push_back(j_output_ignore);
    output_ignore_sets.push_back(k_output_ignore);
    input_ignore_sets.push_back(i_input_ignore);
    input_ignore_sets.push_back(j_input_ignore);
    input_ignore_sets.push_back(k_input_ignore);

    //Mess up the i loop for A
    input_indices_sets[0][0] = 1; //should be zero


    block_contract2_kernel<double> bc2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets); 


    dim_list output_dims_1; //C
    output_dims_1.push_back(2);
    output_dims_1.push_back(3);

    dim_list input_dims_1; //A
    input_dims_1.push_back(2);
    input_dims_1.push_back(4);

    dim_list input_dims_2; //B
    input_dims_2.push_back(4);
    input_dims_2.push_back(3);


    sequence<1,double*> output_ptrs(test_output_block);
    sequence<2,const double*> input_ptrs(test_input_block_1);
    input_ptrs[1] = test_input_block_2;

    sequence<1, dim_list> output_dims(output_dims_1);
    sequence<2, dim_list> input_dims;
    input_dims[0] = input_dims_1;
    input_dims[1] = input_dims_2;

    bool threw_exception = false;
    try
    {
        bc2k(output_ptrs,input_ptrs,output_dims,input_dims);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel::operator(...) did not throw exception when not matching indices given");
    }
}

//Should throw an exception due to passing a dimension list of the wrong order
void block_kernels_test::test_block_contract2_kernel_2d_wrong_dim_order() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_wrong_dim_order()";

    //C_ij = \sum_k A_ik B_kj
    //dimensions: i = 2,j = 3,k = 4
    //Just need dummy data for this test
    double test_output_block[6]; //C
    double test_input_block_1[8]; //A 
    double test_input_block_2[12]; //B
    
    //Initialize the indices touched by each loop
    std::vector< sequence<1,size_t> > output_indices_sets;
    std::vector< sequence<2,size_t> > input_indices_sets;
    std::vector< sequence<1,bool> > output_ignore_sets;
    std::vector< sequence<2,bool> > input_ignore_sets;

    //i loop
    sequence<1,size_t> i_output_indices(0);
    sequence<2,size_t> i_input_indices(0); //B is ignored

    sequence<1,bool> i_output_ignore(false);
    sequence<2,bool> i_input_ignore(false);
    i_input_ignore[1] = true;

    //j loop
    sequence<1,size_t> j_output_indices(1);
    sequence<2,size_t> j_input_indices(1); //A is ignored

    sequence<1,bool> j_output_ignore(false); 
    sequence<2,bool> j_input_ignore(false);
    j_input_ignore[0] = true;

    //k loop
    sequence<1,size_t> k_output_indices;
    sequence<2,size_t> k_input_indices(1);
    k_input_indices[1] = 0;

    sequence<1,bool> k_output_ignore(true);
    sequence<2,bool> k_input_ignore(false);


    output_indices_sets.push_back(i_output_indices);
    output_indices_sets.push_back(j_output_indices);
    output_indices_sets.push_back(k_output_indices);
    input_indices_sets.push_back(i_input_indices);
    input_indices_sets.push_back(j_input_indices);
    input_indices_sets.push_back(k_input_indices);

    output_ignore_sets.push_back(i_output_ignore);
    output_ignore_sets.push_back(j_output_ignore);
    output_ignore_sets.push_back(k_output_ignore);
    input_ignore_sets.push_back(i_input_ignore);
    input_ignore_sets.push_back(j_input_ignore);
    input_ignore_sets.push_back(k_input_ignore);

    block_contract2_kernel<double> bc2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets); 

    dim_list output_dims_1; //C
    output_dims_1.push_back(2);
    output_dims_1.push_back(3);

    dim_list input_dims_1; //A
    input_dims_1.push_back(2);
    input_dims_1.push_back(4);
    //Mess this up on purpose - add extra entry
    input_dims_1.push_back(2);
    input_dims_1.push_back(2);
    
    dim_list input_dims_2; //B
    input_dims_2.push_back(4);
    input_dims_2.push_back(3);


    sequence<1,double*> output_ptrs(test_output_block);
    sequence<2,const double*> input_ptrs(test_input_block_1);
    input_ptrs[1] = test_input_block_2;

    sequence<1, dim_list> output_dims(output_dims_1);
    sequence<2, dim_list> input_dims;
    input_dims[0] = input_dims_1;
    input_dims[1] = input_dims_2;

    bool threw_exception = false;
    try
    {
        bc2k(output_ptrs,input_ptrs,output_dims,input_dims);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "block_contract2_kernel::operator(...) did not throw exception when wrong order dimension list passed");
    }
}

//Should produce correct matrix multiply output, with matrices in standard order
void block_kernels_test::test_block_contract2_kernel_2d_ip_pj() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_ip_pj()";

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
    
    //Initialize the indices touched by each loop
    std::vector< sequence<1,size_t> > output_indices_sets;
    std::vector< sequence<2,size_t> > input_indices_sets;
    std::vector< sequence<1,bool> > output_ignore_sets;
    std::vector< sequence<2,bool> > input_ignore_sets;

    //i loop
    sequence<1,size_t> i_output_indices(0);
    sequence<2,size_t> i_input_indices(0); //B is ignored

    sequence<1,bool> i_output_ignore(false);
    sequence<2,bool> i_input_ignore(false);
    i_input_ignore[1] = true;

    //j loop
    sequence<1,size_t> j_output_indices(1);
    sequence<2,size_t> j_input_indices(1); //A is ignored

    sequence<1,bool> j_output_ignore(false); 
    sequence<2,bool> j_input_ignore(false);
    j_input_ignore[0] = true;

    //k loop
    sequence<1,size_t> k_output_indices;
    sequence<2,size_t> k_input_indices(1);
    k_input_indices[1] = 0;

    sequence<1,bool> k_output_ignore(true);
    sequence<2,bool> k_input_ignore(false);


    output_indices_sets.push_back(i_output_indices);
    output_indices_sets.push_back(j_output_indices);
    output_indices_sets.push_back(k_output_indices);
    input_indices_sets.push_back(i_input_indices);
    input_indices_sets.push_back(j_input_indices);
    input_indices_sets.push_back(k_input_indices);

    output_ignore_sets.push_back(i_output_ignore);
    output_ignore_sets.push_back(j_output_ignore);
    output_ignore_sets.push_back(k_output_ignore);
    input_ignore_sets.push_back(i_input_ignore);
    input_ignore_sets.push_back(j_input_ignore);
    input_ignore_sets.push_back(k_input_ignore);

    block_contract2_kernel<double> bc2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets); 


    dim_list output_dims_1; //C
    output_dims_1.push_back(2);
    output_dims_1.push_back(3);

    dim_list input_dims_1; //A
    input_dims_1.push_back(2);
    input_dims_1.push_back(4);

    dim_list input_dims_2; //B
    input_dims_2.push_back(4);
    input_dims_2.push_back(3);


    sequence<1,double*> output_ptrs(test_output_block);
    sequence<2,const double*> input_ptrs(test_input_block_1);
    input_ptrs[1] = test_input_block_2;

    sequence<1, dim_list> output_dims(output_dims_1);
    sequence<2, dim_list> input_dims;
    input_dims[0] = input_dims_1;
    input_dims[1] = input_dims_2;

    bc2k(output_ptrs,input_ptrs,output_dims,input_dims);

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
void block_kernels_test::test_block_contract2_kernel_2d_ip_jp() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_ip_jp()";

    //C_ij = \sum_k A_ik B_kj
    //dimensions: i = 2,j = 3,k = 4
    //Just need dummy data for this test
    
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
    
    //Initialize the indices touched by each loop
    std::vector< sequence<1,size_t> > output_indices_sets;
    std::vector< sequence<2,size_t> > input_indices_sets;
    std::vector< sequence<1,bool> > output_ignore_sets;
    std::vector< sequence<2,bool> > input_ignore_sets;

    //i loop
    sequence<1,size_t> i_output_indices(0);
    sequence<2,size_t> i_input_indices(0); //B is ignored

    sequence<1,bool> i_output_ignore(false);
    sequence<2,bool> i_input_ignore(false);
    i_input_ignore[1] = true;

    //j loop
    sequence<1,size_t> j_output_indices(1);
    sequence<2,size_t> j_input_indices(0); //A is ignored

    sequence<1,bool> j_output_ignore(false); 
    sequence<2,bool> j_input_ignore(false);
    j_input_ignore[0] = true;

    //k loop
    sequence<1,size_t> k_output_indices;
    sequence<2,size_t> k_input_indices(1);

    sequence<1,bool> k_output_ignore(true);
    sequence<2,bool> k_input_ignore(false);


    output_indices_sets.push_back(i_output_indices);
    output_indices_sets.push_back(j_output_indices);
    output_indices_sets.push_back(k_output_indices);
    input_indices_sets.push_back(i_input_indices);
    input_indices_sets.push_back(j_input_indices);
    input_indices_sets.push_back(k_input_indices);

    output_ignore_sets.push_back(i_output_ignore);
    output_ignore_sets.push_back(j_output_ignore);
    output_ignore_sets.push_back(k_output_ignore);
    input_ignore_sets.push_back(i_input_ignore);
    input_ignore_sets.push_back(j_input_ignore);
    input_ignore_sets.push_back(k_input_ignore);

    block_contract2_kernel<double> bc2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets); 


    dim_list output_dims_1; //C
    output_dims_1.push_back(2);
    output_dims_1.push_back(3);

    dim_list input_dims_1; //A
    input_dims_1.push_back(2);
    input_dims_1.push_back(4);

    dim_list input_dims_2; //B
    input_dims_2.push_back(3);
    input_dims_2.push_back(4);


    sequence<1,double*> output_ptrs(test_output_block);
    sequence<2,const double*> input_ptrs(test_input_block_1);
    input_ptrs[1] = test_input_block_2;

    sequence<1, dim_list> output_dims(output_dims_1);
    sequence<2, dim_list> input_dims;
    input_dims[0] = input_dims_1;
    input_dims[1] = input_dims_2;

    bc2k(output_ptrs,input_ptrs,output_dims,input_dims);

    for(int i = 0; i < 6; ++i)
    {
        if(test_output_block[i] != correct_output_block[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_contract2_kernel::operator(...) did not produce correct result");
        }
    }
}

void block_kernels_test::test_block_contract2_kernel_2d_pi_pj() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_pi_pj()";

    //C_ij = \sum_k A_ik B_kj
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
    
    //Initialize the indices touched by each loop
    std::vector< sequence<1,size_t> > output_indices_sets;
    std::vector< sequence<2,size_t> > input_indices_sets;
    std::vector< sequence<1,bool> > output_ignore_sets;
    std::vector< sequence<2,bool> > input_ignore_sets;

    //i loop
    sequence<1,size_t> i_output_indices(0);
    sequence<2,size_t> i_input_indices(1); //B is ignored

    sequence<1,bool> i_output_ignore(false);
    sequence<2,bool> i_input_ignore(false);
    i_input_ignore[1] = true;

    //j loop
    sequence<1,size_t> j_output_indices(1);
    sequence<2,size_t> j_input_indices(1); //A is ignored

    sequence<1,bool> j_output_ignore(false); 
    sequence<2,bool> j_input_ignore(false);
    j_input_ignore[0] = true;

    //k loop
    sequence<1,size_t> k_output_indices;
    sequence<2,size_t> k_input_indices(0);
    k_input_indices[1] = 0;

    sequence<1,bool> k_output_ignore(true);
    sequence<2,bool> k_input_ignore(false);


    output_indices_sets.push_back(i_output_indices);
    output_indices_sets.push_back(j_output_indices);
    output_indices_sets.push_back(k_output_indices);
    input_indices_sets.push_back(i_input_indices);
    input_indices_sets.push_back(j_input_indices);
    input_indices_sets.push_back(k_input_indices);

    output_ignore_sets.push_back(i_output_ignore);
    output_ignore_sets.push_back(j_output_ignore);
    output_ignore_sets.push_back(k_output_ignore);
    input_ignore_sets.push_back(i_input_ignore);
    input_ignore_sets.push_back(j_input_ignore);
    input_ignore_sets.push_back(k_input_ignore);

    block_contract2_kernel<double> bc2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets); 


    dim_list output_dims_1; //C
    output_dims_1.push_back(2);
    output_dims_1.push_back(3);

    dim_list input_dims_1; //A
    input_dims_1.push_back(4);
    input_dims_1.push_back(2);

    dim_list input_dims_2; //B
    input_dims_2.push_back(4);
    input_dims_2.push_back(3);


    sequence<1,double*> output_ptrs(test_output_block);
    sequence<2,const double*> input_ptrs(test_input_block_1);
    input_ptrs[1] = test_input_block_2;

    sequence<1, dim_list> output_dims(output_dims_1);
    sequence<2, dim_list> input_dims;
    input_dims[0] = input_dims_1;
    input_dims[1] = input_dims_2;

    bc2k(output_ptrs,input_ptrs,output_dims,input_dims);

    for(int i = 0; i < 6; ++i)
    {
        if(test_output_block[i] != correct_output_block[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_contract2_kernel::operator(...) did not produce correct result");
        }
    }
}

void block_kernels_test::test_block_contract2_kernel_2d_pi_jp() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_2d_pi_jp()";

    //C_ij = \sum_k A_ik B_kj
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
    
    //Initialize the indices touched by each loop
    std::vector< sequence<1,size_t> > output_indices_sets;
    std::vector< sequence<2,size_t> > input_indices_sets;
    std::vector< sequence<1,bool> > output_ignore_sets;
    std::vector< sequence<2,bool> > input_ignore_sets;

    //i loop
    sequence<1,size_t> i_output_indices(0);
    sequence<2,size_t> i_input_indices(1); //B is ignored

    sequence<1,bool> i_output_ignore(false);
    sequence<2,bool> i_input_ignore(false);
    i_input_ignore[1] = true;

    //j loop
    sequence<1,size_t> j_output_indices(1);
    sequence<2,size_t> j_input_indices(0); //A is ignored

    sequence<1,bool> j_output_ignore(false); 
    sequence<2,bool> j_input_ignore(false);
    j_input_ignore[0] = true;

    //k loop
    sequence<1,size_t> k_output_indices;
    sequence<2,size_t> k_input_indices(0);
    k_input_indices[1] = 1;

    sequence<1,bool> k_output_ignore(true);
    sequence<2,bool> k_input_ignore(false);


    output_indices_sets.push_back(i_output_indices);
    output_indices_sets.push_back(j_output_indices);
    output_indices_sets.push_back(k_output_indices);
    input_indices_sets.push_back(i_input_indices);
    input_indices_sets.push_back(j_input_indices);
    input_indices_sets.push_back(k_input_indices);

    output_ignore_sets.push_back(i_output_ignore);
    output_ignore_sets.push_back(j_output_ignore);
    output_ignore_sets.push_back(k_output_ignore);
    input_ignore_sets.push_back(i_input_ignore);
    input_ignore_sets.push_back(j_input_ignore);
    input_ignore_sets.push_back(k_input_ignore);

    block_contract2_kernel<double> bc2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets); 


    dim_list output_dims_1; //C
    output_dims_1.push_back(2);
    output_dims_1.push_back(3);

    dim_list input_dims_1; //A
    input_dims_1.push_back(4);
    input_dims_1.push_back(2);

    dim_list input_dims_2; //B
    input_dims_2.push_back(3);
    input_dims_2.push_back(4);


    sequence<1,double*> output_ptrs(test_output_block);
    sequence<2,const double*> input_ptrs(test_input_block_1);
    input_ptrs[1] = test_input_block_2;

    sequence<1, dim_list> output_dims(output_dims_1);
    sequence<2, dim_list> input_dims;
    input_dims[0] = input_dims_1;
    input_dims[1] = input_dims_2;

    bc2k(output_ptrs,input_ptrs,output_dims,input_dims);

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
    static const char *test_name = "block_kernels_test::test_block_permute_kernel_3d_2d()";

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

    
    //Initialize the indices touched by each loop
    std::vector< sequence<1,size_t> > output_indices_sets;
    std::vector< sequence<2,size_t> > input_indices_sets;
    std::vector< sequence<1,bool> > output_ignore_sets;
    std::vector< sequence<2,bool> > input_ignore_sets;

    //i loop
    sequence<1,size_t> i_output_indices(0);
    sequence<2,size_t> i_input_indices(0); //B is ignored

    sequence<1,bool> i_output_ignore(false);
    sequence<2,bool> i_input_ignore(false);
    i_input_ignore[1] = true;

    //j loop
    sequence<1,size_t> j_output_indices(1);
    sequence<2,size_t> j_input_indices(1); //B is ignored

    sequence<1,bool> j_output_ignore(false); 
    sequence<2,bool> j_input_ignore(false);
    j_input_ignore[1] = true;

    //k loop
    sequence<1,size_t> k_output_indices;
    sequence<2,size_t> k_input_indices(2);
    k_input_indices[1] = 0;

    sequence<1,bool> k_output_ignore(true);
    sequence<2,bool> k_input_ignore(false);

    //l loop
    sequence<1,size_t> l_output_indices(2);
    sequence<2,size_t> l_input_indices(1); //A is ignored

    sequence<1,bool> l_output_ignore(false);
    sequence<2,bool> l_input_ignore(false);
    l_input_ignore[0] = true;


    output_indices_sets.push_back(i_output_indices);
    output_indices_sets.push_back(j_output_indices);
    output_indices_sets.push_back(l_output_indices);
    output_indices_sets.push_back(k_output_indices);
    input_indices_sets.push_back(i_input_indices);
    input_indices_sets.push_back(j_input_indices);
    input_indices_sets.push_back(l_input_indices);
    input_indices_sets.push_back(k_input_indices);

    output_ignore_sets.push_back(i_output_ignore);
    output_ignore_sets.push_back(j_output_ignore);
    output_ignore_sets.push_back(l_output_ignore);
    output_ignore_sets.push_back(k_output_ignore);
    input_ignore_sets.push_back(i_input_ignore);
    input_ignore_sets.push_back(j_input_ignore);
    input_ignore_sets.push_back(l_input_ignore);
    input_ignore_sets.push_back(k_input_ignore);

    block_contract2_kernel<double> bc2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets); 


    dim_list output_dims_1; //C
    output_dims_1.push_back(2);
    output_dims_1.push_back(3);
    output_dims_1.push_back(5);

    dim_list input_dims_1; //A
    input_dims_1.push_back(2);
    input_dims_1.push_back(3);
    input_dims_1.push_back(4);

    dim_list input_dims_2; //B
    input_dims_2.push_back(4);
    input_dims_2.push_back(5);


    sequence<1,double*> output_ptrs(test_output_block);
    sequence<2,const double*> input_ptrs(test_input_block_1);
    input_ptrs[1] = test_input_block_2;

    sequence<1, dim_list> output_dims(output_dims_1);
    sequence<2, dim_list> input_dims;
    input_dims[0] = input_dims_1;
    input_dims[1] = input_dims_2;

    bc2k(output_ptrs,input_ptrs,output_dims,input_dims);

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

    //i loop
    sequence<1,size_t> i_output_indices(0);
    sequence<2,size_t> i_input_indices(0); //B is ignored
    sequence<1,bool> i_output_ignore(false);
    sequence<2,bool> i_input_ignore(false);
    i_input_ignore[1] = true;

    //j loop
    sequence<1,size_t> j_output_indices;
    sequence<2,size_t> j_input_indices(1);
    sequence<1,bool> j_output_ignore(true); 
    sequence<2,bool> j_input_ignore(false);

    //k loop
    sequence<1,size_t> k_output_indices;
    sequence<2,size_t> k_input_indices(2);
    sequence<1,bool> k_output_ignore(true);
    sequence<2,bool> k_input_ignore(false);

    //l loop
    sequence<1,size_t> l_output_indices(1);
    sequence<2,size_t> l_input_indices(0); //A is ignored
    sequence<1,bool> l_output_ignore(false);
    sequence<2,bool> l_input_ignore(false);
    l_input_ignore[0] = true;

    //Kernel 
    std::vector< sequence<1,size_t> > output_indices_sets;
    std::vector< sequence<2,size_t> > input_indices_sets;
    std::vector< sequence<1,bool> > output_ignore_sets;
    std::vector< sequence<2,bool> > input_ignore_sets;
    output_indices_sets.push_back(i_output_indices);
    output_indices_sets.push_back(l_output_indices);
    output_indices_sets.push_back(j_output_indices);
    output_indices_sets.push_back(k_output_indices);
    input_indices_sets.push_back(i_input_indices);
    input_indices_sets.push_back(l_input_indices);
    input_indices_sets.push_back(j_input_indices);
    input_indices_sets.push_back(k_input_indices);

    output_ignore_sets.push_back(i_output_ignore);
    output_ignore_sets.push_back(l_output_ignore);
    output_ignore_sets.push_back(j_output_ignore);
    output_ignore_sets.push_back(k_output_ignore);
    input_ignore_sets.push_back(i_input_ignore);
    input_ignore_sets.push_back(l_input_ignore);
    input_ignore_sets.push_back(j_input_ignore);
    input_ignore_sets.push_back(k_input_ignore);
    block_contract2_kernel<double> bc2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets); 

    //Pointers
    sequence<1,double*> output_ptrs(C_arr);
    sequence<2,const double*> input_ptrs(A_arr);
    input_ptrs[1] = B_arr;

    //Dims
    dim_list C_dims;
    C_dims.push_back(2);
    C_dims.push_back(5);

    dim_list A_dims;
    A_dims.push_back(2);
    A_dims.push_back(3);
    A_dims.push_back(4);

    dim_list B_dims;
    B_dims.push_back(5);
    B_dims.push_back(3);
    B_dims.push_back(4);


    sequence<1, dim_list> output_dims(C_dims);
    sequence<2, dim_list> input_dims;
    input_dims[0] = A_dims;
    input_dims[1] = B_dims;


    bc2k(output_ptrs,input_ptrs,output_dims,input_dims);

    for(int i = 0; i < 10; ++i)
    {
        if(C_arr[i] != C_correct_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_contract2_kernel::operator(...) did not produce correct result");
        }
    }
}

void block_kernels_test::test_block_subtract_kernel_2d_2d() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_block_subtract_kernel_2d_2d()";

    double A_arr[20] = {1,2,3,4,5,
                        6,7,8,9,10,
                        11,12,13,14,15,
                        16,17,18,19,29};

    //Row major
    double B_arr[20] = {21,26,31,36,41,
                        22,27,32,37,42,
                        23,28,33,38,43,
                        24,29,34,39,44};

    double C_arr[20] = {0};

    //Row major
    double C_correct_arr[20] = {20,24,28,32,36,
                                16,20,24,28,32,
                                12,16,20,24,28,
                                8,12,16,20,15};

    std::vector< sequence<1,size_t> > output_indices_sets(2);
    output_indices_sets[0] = sequence<1,size_t>(0);
    output_indices_sets[1] = sequence<1,size_t>(1);

    std::vector< sequence<2,size_t> > input_indices_sets(2);
    input_indices_sets[0][0] = 0;
    input_indices_sets[0][1] = 0;
    input_indices_sets[1][0] = 1;
    input_indices_sets[1][1] = 1;

    std::vector< sequence<1,bool> > output_ignore_sets(2,sequence<1,bool>(false));
    std::vector< sequence<2,bool> > input_ignore_sets(2,sequence<2,bool>(false));

    block_subtract2_kernel<double> bs2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets);

    sequence<1,double*> output_ptrs(C_arr); 
    sequence<2,const double*> input_ptrs(B_arr); 
    input_ptrs[1] = A_arr;
    sequence<1,dim_list > output_dims;
    output_dims[0].push_back(4);
    output_dims[0].push_back(5);
    sequence<2,dim_list> input_dims;
    input_dims[0].push_back(4);
    input_dims[0].push_back(5);
    input_dims[1].push_back(4);
    input_dims[1].push_back(5);
    bs2k(output_ptrs,input_ptrs,output_dims,input_dims);

    for(int i = 0; i < 20; ++i)
    {
        if(C_arr[i] != C_correct_arr[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "block_subtract2_kernel::operator(...) did not produce correct result");
        }
    }
}

#if 0
void block_kernels_test::test_direct_block_subtract_kernel_2d_2d() throw(libtest::test_exception)
{
    static const char *test_name = "block_kernels_test::test_direct_block_subtract_kernel_2d_2d()";

    double A_arr[20] = {1,2,3,4,5,
                        6,7,8,9,10,
                        11,12,13,14,15,
                        16,17,18,19,29};

    //Row major
    double B_arr[20] = {21,26,31,36,41,
                        22,27,32,37,42,
                        23,28,33,38,43,
                        24,29,34,39,44};


    //Row major
    double C_correct_arr[20] = {20,24,28,32,36,
                                16,20,24,28,32,
                                12,16,20,24,28,
                                8,12,16,20,15};

    //First, generate the kernel for basic subtraction, that will be used to generate our direct intermediate
    std::vector< sequence<1,size_t> > output_indices_sets(2);
    output_indices_sets[0] = sequence<1,size_t>(0);
    output_indices_sets[1] = sequence<1,size_t>(1);

    std::vector< sequence<2,size_t> > input_indices_sets(2);
    input_indices_sets[0][0] = 0;
    input_indices_sets[0][1] = 0;
    input_indices_sets[1][0] = 1;
    input_indices_sets[1][1] = 1;

    std::vector< sequence<1,bool> > output_ignore_sets(2,sequence<1,bool>(false));
    std::vector< sequence<2,bool> > input_ignore_sets(2,sequence<2,bool>(false));

    block_subtract2_kernel<double> bs2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets);

    sequence<1,double*> output_ptrs(C_arr); 
    sequence<2,const double*> input_ptrs(B_arr); 
    input_ptrs[1] = A_arr;
    sequence<1,dim_list > output_dims;
    output_dims[0].push_back(4);
    output_dims[0].push_back(5);
    sequence<2,dim_list> input_dims;
    input_dims[0].push_back(4);
    input_dims[0].push_back(5);
    input_dims[1].push_back(4);
    input_dims[1].push_back(5);
    bs2k(output_ptrs,input_ptrs,output_dims,input_dims);

    //1 block for simplicity
    sparse_bispace<1> spb_i(4);
    sparse_bispace<1> spb_j(5);
    sparse_bispace<2> spb_A = spb_i | spb_j;
    sparse_btensor<2> A(spb_A,A_arr);
    sparse_btensor<2> C(spb_A);

    //We execute C - A, with C being formed direct from B and A. The final result is stored in D
    letter i,j;
    direct_block_kernel<1,2> d_kern(bs2k,0,B(i|j) - A(i|j),C(i|j));

    //Correct answer
    double D_correct_arr[20] = {19,22,25,28,31,
                                10,13,16,19,22,
                                1,4,7,10,13,
                                -8,-5,-2,1,-14};


    sparse_btensor<2> D_correct(spb_A,D_correct_arr)
    if(D != D_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "direct_block_kernel::operator(...) did not produce correct result");
    }
}
#endif

} // namespace libtensor
