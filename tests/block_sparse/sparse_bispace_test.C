#include <libtensor/block_sparse/sparse_bispace.h> 
#include <libtensor/core/out_of_bounds.h>
#include <vector>
#include "sparse_bispace_test.h"

namespace libtensor {

void sparse_bispace_test::perform() throw(libtest::test_exception) {

        test_get_dim();

        test_split_not_strictly_increasing();
        test_split_not_strictly_increasing_two_calls();
        test_split_gt_upper_bound();
        test_split_zero_size();

        test_equality_true();
        test_equality_false_diff_dims();
        test_equality_false_diff_splits();
        test_nd_equality_true();

        test_get_block_abs_index_one_block();
        test_get_block_abs_index_two_block();
        test_get_block_abs_index_gt_upper_bound();

        test_get_block_size_one_block();
        test_get_block_size_two_block();
        test_get_block_size_gt_upper_bound();

        test_nd_subscript_operator_gt_upper_bound();

        test_nd_bar_operator_both_operands_1d();
        test_nd_bar_operator_both_operands_2d();

        test_get_nnz_dense(); 

        test_permute_2d_10();
        test_permute_3d_dense_sparse_021();
        test_permute_3d_fully_sparse_210();

        test_contract_3d_dense();
        test_contract_3d_sparse_2();
        test_contract_3d_sparse_2_nnz();
        test_contract_3d_sparse_destroy_all_sparsity();

        test_fuse_2d_2d();
        test_fuse_3d_3d_no_overlap();
        test_fuse_3d_3d_invalid_no_match();

        test_get_n_index_groups();
        test_get_index_group_offset();
        test_get_index_group_order();
        test_get_index_group_dim();
        test_get_index_group_containing_subspace();

        test_get_nnz_2d_sparsity();
        test_get_nnz_3d_dense_sparse();
        test_get_nnz_3d_sparse_dense();
        test_get_nnz_3d_fully_sparse();

        test_equality_false_sparsity_2d();
        test_equality_true_sparsity_2d();
}

//TEST FIXTURES
namespace {     

class index_groups_test_f {
private:
    static sparse_bispace<7> init_bispace()
    {
        //Need 5 blocks of size 2 
        sparse_bispace<1> spb_0(10);
        idx_list split_points_0;
        for(size_t i = 2; i < spb_0.get_dim(); i += 2)
        {
            split_points_0.push_back(i);
        } 
        spb_0.split(split_points_0);


        //Need 6 blocks of size 3
        sparse_bispace<1> spb_1(18);
        idx_list split_points_1;
        for(size_t i = 3; i < spb_1.get_dim(); i += 3)
        {
            split_points_1.push_back(i);
        }
        spb_1.split(split_points_1);

        size_t key_0_arr_0[2] = {2,1}; //offset 0
        size_t key_1_arr_0[2] = {3,2}; //offset 6
        size_t key_2_arr_0[2] = {4,3}; //offset 12
        size_t key_3_arr_0[2] = {5,4}; //offset 18
        std::vector< sequence<2,size_t> > sig_blocks_0(4);
        for(size_t i = 0; i < 2; ++i) sig_blocks_0[0][i] = key_0_arr_0[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_0[1][i] = key_1_arr_0[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_0[2][i] = key_2_arr_0[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_0[3][i] = key_3_arr_0[i];

        size_t key_0_arr_1[2] = {1,4}; //offset 0
        size_t key_1_arr_1[2] = {2,1}; //offset 9
        size_t key_2_arr_1[2] = {3,3}; //offset 18
        std::vector< sequence<2,size_t> > sig_blocks_1(3);
        for(size_t i = 0; i < 2; ++i) sig_blocks_1[0][i] = key_0_arr_1[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_1[1][i] = key_1_arr_1[i];
        for(size_t i = 0; i < 2; ++i) sig_blocks_1[2][i] = key_2_arr_1[i];

        return spb_0 | spb_0 | spb_1 % spb_0 << sig_blocks_0 | spb_0 | spb_1 % spb_1 << sig_blocks_1;
    }
public:
    sparse_bispace<7> bispace;
    index_groups_test_f() : bispace(init_bispace()) {}
};

} // namespace unnamed

/* Return the correct value for the dimension of the sparse_bispace
 *
 */
void sparse_bispace_test::test_get_dim() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_dim()";
    sparse_bispace<1> spb(5);
    if(spb.get_dim() != 5)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::get_dim(...) returned incorrect value");
    } 
}

void sparse_bispace_test::test_get_n_blocks() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_n_blocks()";
    sparse_bispace<1> spb(5);
    std::vector<size_t> split_points; 
    split_points.push_back(1);
    split_points.push_back(3);
    spb.split(split_points);

    if(spb.get_n_blocks() != 3)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::get_n_blocks(...) returned incorrect value");
    } 
}



/* Should throw out_of_bounds when split points are passed with an order
 * that is not strictly increasing.
 *
 */
void sparse_bispace_test::test_split_not_strictly_increasing() throw(libtest::test_exception) {

    static const char *test_name = "sparse_bispace_test::test_split_not_strictly_increasing()";

    bool threw_exception = false;
    sparse_bispace<1> spb(5);
    std::vector<size_t> split_points;
    split_points.push_back(3);
    split_points.push_back(1);
    try
    {
        spb.split(split_points);
    }
    catch(out_of_bounds&)
    {
        threw_exception = true;
    }

    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::split(...) did not throw exception when split points not strictly increasing");
    }
}

/* Should throw out_of_bounds when split points are passed with an order
 * that is not strictly increasing, over two consecutive calls
 *
 */
void sparse_bispace_test::test_split_not_strictly_increasing_two_calls() throw(libtest::test_exception) {

    static const char *test_name = "sparse_bispace_test::test_split_not_strictly_increasing_two_calls()";

    bool threw_exception = false;
    sparse_bispace<1> spb(5);
    std::vector<size_t> split_points;
    split_points.push_back(3);
    spb.split(split_points);
    split_points[0] = 1;
    try
    {
        spb.split(split_points);
    }
    catch(out_of_bounds&)
    {
        threw_exception = true;
    }

    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::split(...) did not throw exception when split points not strictly increasing");
    }
}

/* Should throw out_of_bounds when a split_point that exceeds the dimension of the space is passed 
 */
void sparse_bispace_test::test_split_gt_upper_bound() throw(libtest::test_exception) {

    static const char *test_name = "sparse_bispace_test::test_split_gt_upper_bound()";

    bool threw_exception = false;
    sparse_bispace<1> spb(5);
    std::vector<size_t> split_points;
    split_points.push_back(5);
    try
    {
        spb.split(split_points);
    }
    catch(out_of_bounds&)
    {
        threw_exception = true;
    }

    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::split(...) did not throw exception when split point index greater than max index value");
    }
}

/* Should throw out_of_bounds when zero split_points are specified
 */
void sparse_bispace_test::test_split_zero_size() throw(libtest::test_exception) {

    static const char *test_name = "sparse_bispace_test::test_split_zero_size()";

    bool threw_exception = false;
    sparse_bispace<1> spb(5);
    std::vector<size_t> split_points;
    try
    {
        spb.split(split_points);
    }
    catch(out_of_bounds&)
    {
        threw_exception = true;
    }

    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::split(...) did not throw exception when zero points specified");
    }
}

void sparse_bispace_test::test_equality_true() throw(libtest::test_exception) { 

    static const char *test_name = "sparse_bispace_test::test_equality_true()";

    sparse_bispace<1> spb_1(5);
    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points;
    split_points.push_back(3);
    split_points.push_back(4);
    spb_1.split(split_points); 
    spb_2.split(split_points); 

    if(!(spb_1 == spb_2)) 
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::operator==(...) returned false");

    }
}

void sparse_bispace_test::test_equality_false_diff_dims() throw(libtest::test_exception) {

    static const char *test_name = "sparse_bispace_test::test_equality_false_diff_dims()";

    sparse_bispace<1> spb_1(5);
    sparse_bispace<1> spb_2(6);
    std::vector<size_t> split_points;
    split_points.push_back(3);
    split_points.push_back(4);
    spb_1.split(split_points); 
    spb_2.split(split_points); 

    if(spb_1 == spb_2) 
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::operator==(...) returned true");
    }
}

void sparse_bispace_test::test_equality_false_diff_splits() throw(libtest::test_exception) {

    static const char *test_name = "sparse_bispace_test::test_equality_false_diff_dims()";

    sparse_bispace<1> spb_1(5);
    sparse_bispace<1> spb_2(5);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(3);
    split_points_1.push_back(4);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    split_points_2.push_back(4);
    spb_1.split(split_points_1); 
    spb_2.split(split_points_2); 

    if(spb_1 == spb_2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::operator==(...) returned true");
    }
}

/* Tests equality operator for multidimensional block index spaces
 *
 */
void sparse_bispace_test::test_nd_equality_true() throw(libtest::test_exception) {

    static const char *test_name = "sparse_bispace_test::test_nd_equality_true()";

    //First two
    sparse_bispace<1> spb_1(5);
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(1);
    split_points_1.push_back(3);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(6);
    std::vector<size_t> split_points_2; 
    split_points_2.push_back(2);
    split_points_2.push_back(5);
    spb_2.split(split_points_2);

    sparse_bispace<2> spb_3 = spb_1 | spb_2; 

    //Identical to the two above - equality should indicate this
    sparse_bispace<1> spb_4(5);
    std::vector<size_t> split_points_4; 
    split_points_4.push_back(1);
    split_points_4.push_back(3);
    spb_4.split(split_points_4);

    sparse_bispace<1> spb_5(6);
    std::vector<size_t> split_points_5; 
    split_points_5.push_back(2);
    split_points_5.push_back(5);
    spb_5.split(split_points_5);

    sparse_bispace<2> spb_6 = spb_4 | spb_5; 

    if(!(spb_3 == spb_6))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::operator==(...) returned false");
    }
}

/* Should return zero
 */
void sparse_bispace_test::test_get_block_abs_index_one_block() throw(libtest::test_exception) {
    static const char *test_name = "sparse_bispace_test::test_get_block_abs_index_one_block()";
    sparse_bispace<1> spb(5);
    if(! (spb.get_block_abs_index(0) == 0))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::get_block_abs_index(0) did not return zero");
    }
}

/* Should return 4
 */
void sparse_bispace_test::test_get_block_abs_index_two_block() throw(libtest::test_exception) {
    static const char *test_name = "sparse_bispace_test::test_get_block_abs_index_one_block()";
    sparse_bispace<1> spb(5);
    std::vector<size_t> split_points;
    split_points.push_back(4);
    spb.split(split_points);
    if(! (spb.get_block_abs_index(1) == 4))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::get_block_abs_index(1) did not return 4");
    }
}

 /* Should throw out_of_bounds
  */ 
void sparse_bispace_test::test_get_block_abs_index_gt_upper_bound() throw(libtest::test_exception) {
    static const char *test_name = "sparse_bispace_test::test_get_block_abs_index_gt_upper_bound()";
    bool threw_exception = false;
    sparse_bispace<1> spb(5);
    std::vector<size_t> split_points;
    split_points.push_back(4);
    spb.split(split_points);
    try
    {
        spb.get_block_abs_index(2);
    }
    catch(out_of_bounds&)
    {
        threw_exception = true;
    }

    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::get_block_abs_index(...) did not throw exception when when block_idx > (# of blocks - 1)");
    }
}

/* Should return m_dim
 */
void sparse_bispace_test::test_get_block_size_one_block() throw(libtest::test_exception) {

    static const char *test_name = "sparse_bispace_test::test_get_block_size_one_block()";
    sparse_bispace<1> spb(5);
    if(! (spb.get_block_size(0) == 5))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::get_block_size(...) did not throw exception when zero points specified");
    }

}

void sparse_bispace_test::test_get_block_size_two_block() throw(libtest::test_exception) {
    static const char *test_name = "sparse_bispace_test::test_get_block_size_two_block()";
    sparse_bispace<1> spb(5); 
    std::vector<size_t> split_points;
    split_points.push_back(2);
    spb.split(split_points);

    if(! (spb.get_block_size(0) == 2))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::get_block_size(...) returned invalid size for block_idx 0");
    }
    if(! (spb.get_block_size(1) == 3))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::get_block_size(...) returned invalid size for block_idx 1");
    }

}

void sparse_bispace_test::test_get_block_size_gt_upper_bound() throw(libtest::test_exception) {

    static const char *test_name = "sparse_bispace_test::test_get_block_size_gt_upper_bound()";
    bool threw_exception = false;
    sparse_bispace<1> spb(5); 
    std::vector<size_t> split_points;
    split_points.push_back(2);
    spb.split(split_points);

    try
    {
        spb.get_block_size(2);
    }
    catch(out_of_bounds&)
    {
        threw_exception = true;
    }

    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<1>::get_block_size(...) did not throw exception when when block_idx > (# of blocks - 1)");
    }
}

/*
 * sparse_bispace<N>::operator[](...) must throw appropriate exception for out_of_bounds 
 */
void sparse_bispace_test::test_nd_subscript_operator_gt_upper_bound() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_nd_subscript_operator_gt_upper_bound()";
    bool threw_exception = false;
    sparse_bispace<1> spb_1(5);
    sparse_bispace<1> spb_2(6);
    sparse_bispace<2> two_d = spb_1 | spb_2;

    try
    {
        sparse_bispace<1> fail = two_d[2];
    }
    catch(out_of_bounds&)
    {
        threw_exception = true;
    }

    if(! threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::operator[](...) did not throw exception when subscript out of bounds");
    }
}

/*
 * sparse_bispace<N>::operator[](...) must return values equal to initial subspaces passed to operator|
 */
void sparse_bispace_test::test_nd_bar_operator_both_operands_1d() throw(libtest::test_exception) {
    static const char *test_name = "sparse_bispace_test::test_nd_bar_operator_both_operands_1d()";

    sparse_bispace<1> spb_1(5);
    sparse_bispace<1> spb_2(6);

    std::vector<size_t> split_points_1; 
    split_points_1.push_back(1);
    split_points_1.push_back(3);
    spb_1.split(split_points_1);

    std::vector<size_t> split_points_2; 
    split_points_2.push_back(2);
    split_points_2.push_back(5);
    spb_2.split(split_points_2);


    sparse_bispace<2> two_d = spb_1 | spb_2; 

    if(!(two_d[0] == spb_1))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::operator[0](...) did not return spb_1");
    }
    if(!(two_d[1] == spb_2))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::operator[1](...) did not return spb_2");
    }
}

/*
 * sparse_bispace<N>::operator[](...) must return appropriate values
 */
void sparse_bispace_test::test_nd_bar_operator_both_operands_2d() throw(libtest::test_exception) {
    static const char *test_name = "sparse_bispace_test::test_nd_bar_operator_both_operands_1d()";

    sparse_bispace<1> spb_1(5);
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(1);
    split_points_1.push_back(3);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(6);
    std::vector<size_t> split_points_2; 
    split_points_2.push_back(2);
    split_points_2.push_back(5);
    spb_2.split(split_points_2);

    sparse_bispace<1> spb_3(7);
    std::vector<size_t> split_points_3; 
    split_points_3.push_back(4);
    split_points_3.push_back(5);
    spb_3.split(split_points_3);

    sparse_bispace<1> spb_4(8);
    std::vector<size_t> split_points_4; 
    split_points_4.push_back(5);
    split_points_4.push_back(6);
    spb_4.split(split_points_4);

    sparse_bispace<2> two_d_1 = spb_1 | spb_2; 
    sparse_bispace<2> two_d_2 = spb_3 | spb_4; 
    sparse_bispace<4> four_d = two_d_1 | two_d_2;

    if(four_d[0] != spb_1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::operator[0](...) did not return spb_1");
    }
    if(four_d[1] != spb_2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::operator[1](...) did not return spb_2");
    }
    if(four_d[2] != spb_3)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::operator[2](...) did not return spb_3");
    }
    if(four_d[3] != spb_4)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::operator[3](...) did not return spb_4");
    }
}

/* Return the correct value for the # of nonzero elements of a dense sparse bispace
 *
 */
void sparse_bispace_test::test_get_nnz_dense() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_nnz_dense()";
    sparse_bispace<1> spb_1(5);
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(1);
    split_points_1.push_back(3);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(6);
    std::vector<size_t> split_points_2; 
    split_points_2.push_back(2);
    split_points_2.push_back(5);
    spb_2.split(split_points_2);

    sparse_bispace<2> two_d = spb_1 | spb_2;

    if(two_d.get_nnz() != 30)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::get_nnz(...) returned incorrect value");
    } 
}

void sparse_bispace_test::test_permute_2d_10() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_permute_2d_10()";

    sparse_bispace<1> spb_1(5);
    sparse_bispace<1> spb_2(6);

    std::vector<size_t> split_points_1; 
    split_points_1.push_back(1);
    split_points_1.push_back(3);
    spb_1.split(split_points_1);

    std::vector<size_t> split_points_2; 
    split_points_2.push_back(2);
    split_points_2.push_back(5);
    spb_2.split(split_points_2);


    sparse_bispace<2> two_d = spb_1 | spb_2; 

    permutation<2> perm;
    perm.permute(0,1);
    if(two_d.permute(perm) != (spb_2 | spb_1))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::permute(...) returned incorrect value");

    }
}

void sparse_bispace_test::test_permute_3d_dense_sparse_021() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_permute_3d_dense_sparse_021()";

    sparse_bispace<1> spb_1(11);

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     * 3: 2
     */
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    split_points_1.push_back(9);
    spb_1.split(split_points_1);

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     */
    sparse_bispace<1> spb_2(9);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    split_points_2.push_back(5);
    spb_2.split(split_points_2);


    std::vector< sequence<2,size_t> > sig_blocks(4);
    sig_blocks[0][0] = 0; 
    sig_blocks[0][1] = 1;
    sig_blocks[1][0] = 1;
    sig_blocks[1][1] = 2;
    sig_blocks[2][0] = 2;
    sig_blocks[2][1] = 3;
    sig_blocks[3][0] = 3;
    sig_blocks[3][1] = 2;



    sparse_bispace<3> three_d = spb_2 | spb_1 % spb_1 << sig_blocks; 
    
    //Construct the benchmark permuted space
    permutation<3> perm;
    perm.permute(1,2);
    std::vector< sequence<2,size_t> > permuted_sig_blocks(4);
    permuted_sig_blocks[0][0] = 1; 
    permuted_sig_blocks[0][1] = 0;
    permuted_sig_blocks[1][0] = 2;
    permuted_sig_blocks[1][1] = 1;
    permuted_sig_blocks[2][0] = 2;
    permuted_sig_blocks[2][1] = 3;
    permuted_sig_blocks[3][0] = 3;
    permuted_sig_blocks[3][1] = 2;

    sparse_bispace<3> correct_three_d = spb_2 | spb_1 % spb_1 << permuted_sig_blocks; 

    if(three_d.permute(perm) != correct_three_d)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::permute(...) returned incorrect value");
    }
}

void sparse_bispace_test::test_permute_3d_fully_sparse_210() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_permute_3d_fully_sparse_210()";

    sparse_bispace<1> spb_1(11);

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     * 3: 2
     */
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    split_points_1.push_back(9);
    spb_1.split(split_points_1);

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     */
    sparse_bispace<1> spb_2(9);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    split_points_2.push_back(5);
    spb_2.split(split_points_2);


    std::vector< sequence<3,size_t> > sig_blocks(5);

    sig_blocks[0][0] = 0; 
    sig_blocks[0][1] = 0;
    sig_blocks[0][2] = 2;
    sig_blocks[1][0] = 0; 
    sig_blocks[1][1] = 0;
    sig_blocks[1][2] = 3;
    sig_blocks[2][0] = 1;
    sig_blocks[2][1] = 2;
    sig_blocks[2][2] = 2;
    sig_blocks[3][0] = 1;
    sig_blocks[3][1] = 3;
    sig_blocks[3][2] = 1;
    sig_blocks[4][0] = 2;
    sig_blocks[4][1] = 0;
    sig_blocks[4][2] = 1;

    sparse_bispace<3> three_d = spb_2 % spb_1 % spb_1 << sig_blocks; 

    //Make the benchmark
    std::vector< sequence<3,size_t> > permuted_sig_blocks(5);
    permuted_sig_blocks[0][0] = 1;
    permuted_sig_blocks[0][1] = 0;
    permuted_sig_blocks[0][2] = 2;
    permuted_sig_blocks[1][0] = 1;
    permuted_sig_blocks[1][1] = 3;
    permuted_sig_blocks[1][2] = 1;
    permuted_sig_blocks[2][0] = 2; 
    permuted_sig_blocks[2][1] = 0;
    permuted_sig_blocks[2][2] = 0;
    permuted_sig_blocks[3][0] = 2;
    permuted_sig_blocks[3][1] = 2;
    permuted_sig_blocks[3][2] = 1;
    permuted_sig_blocks[4][0] = 3; 
    permuted_sig_blocks[4][1] = 0;
    permuted_sig_blocks[4][2] = 0;
    sparse_bispace<3> correct_three_d = spb_1 % spb_1 % spb_2 << permuted_sig_blocks; 


    permutation<3> perm;
    perm.permute(0,2);
    if(three_d.permute(perm) != correct_three_d)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::permute(...) returned incorrect value");
    }
}

void sparse_bispace_test::test_contract_3d_dense() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_contract_3d_dense()";

    sparse_bispace<1> spb_1(8);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(9);
    std::vector<size_t> split_points_2; 
    split_points_2.push_back(3);
    split_points_2.push_back(6);
    split_points_2.push_back(8);
    spb_2.split(split_points_2);

    sparse_bispace<1> spb_3(10);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(4);
    split_points_3.push_back(7);
    spb_3.split(split_points_3);

    sparse_bispace<3> three_d = spb_1 | spb_2 | spb_3;
    sparse_bispace<2> two_d = three_d.contract(1); 


    sparse_bispace<2> two_d_correct = spb_1 | spb_3; 
    if(two_d != two_d_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::contract(...) returned incorrect value");
    }
}

void sparse_bispace_test::test_contract_3d_sparse_2() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_contract_3d_sparse_2()";

    //Bispaces
    sparse_bispace<1> spb_1(8);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(9);
    std::vector<size_t> split_points_2; 
    split_points_2.push_back(3);
    split_points_2.push_back(6);
    split_points_2.push_back(8);
    spb_2.split(split_points_2);

    sparse_bispace<1> spb_3(10);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(4);
    split_points_3.push_back(7);
    spb_3.split(split_points_3);

    //Sparsity Info
    size_t seq00_arr[3] = {0,0,0};
    size_t seq01_arr[3] = {0,0,2};
    size_t seq02_arr[3] = {0,1,2};
    size_t seq03_arr[3] = {0,2,1};
    size_t seq04_arr[3] = {0,2,2};
    size_t seq05_arr[3] = {0,3,1};
    size_t seq06_arr[3] = {1,0,1};
    size_t seq07_arr[3] = {1,0,2};
    size_t seq08_arr[3] = {1,1,0};
    size_t seq09_arr[3] = {1,2,0};
    size_t seq10_arr[3] = {1,3,1};
    size_t seq11_arr[3] = {2,1,1};
    size_t seq12_arr[3] = {2,2,2};
    size_t seq13_arr[3] = {2,3,0};

    std::vector< sequence<3,size_t> > sig_blocks(14);
    for(size_t i = 0; i < 3; ++i) sig_blocks[0][i] = seq00_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[1][i] = seq01_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[2][i] = seq02_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[3][i] = seq03_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[4][i] = seq04_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[5][i] = seq05_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[6][i] = seq06_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[7][i] = seq07_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[8][i] = seq08_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[9][i] = seq09_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[10][i] = seq10_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[11][i] = seq11_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[12][i] = seq12_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[13][i] = seq13_arr[i];

    sparse_bispace<3> three_d = spb_1 % spb_2 % spb_3 << sig_blocks;
    sparse_bispace<2> two_d = three_d.contract(2); 

    //Correct result
    std::vector< sequence<2,size_t> > contracted_sig_blocks(11);
    size_t contracted_seq00_arr[2] = {0,0};
    size_t contracted_seq01_arr[2] = {0,1};
    size_t contracted_seq02_arr[2] = {0,2};
    size_t contracted_seq03_arr[2] = {0,3};
    size_t contracted_seq04_arr[2] = {1,0};
    size_t contracted_seq05_arr[2] = {1,1};
    size_t contracted_seq06_arr[2] = {1,2};
    size_t contracted_seq07_arr[2] = {1,3};
    size_t contracted_seq08_arr[2] = {2,1};
    size_t contracted_seq09_arr[2] = {2,2};
    size_t contracted_seq10_arr[2] = {2,3};

    for(size_t i = 0; i < 2; ++i) contracted_sig_blocks[0][i] = contracted_seq00_arr[i];
    for(size_t i = 0; i < 2; ++i) contracted_sig_blocks[1][i] = contracted_seq01_arr[i];
    for(size_t i = 0; i < 2; ++i) contracted_sig_blocks[2][i] = contracted_seq02_arr[i];
    for(size_t i = 0; i < 2; ++i) contracted_sig_blocks[3][i] = contracted_seq03_arr[i];
    for(size_t i = 0; i < 2; ++i) contracted_sig_blocks[4][i] = contracted_seq04_arr[i];
    for(size_t i = 0; i < 2; ++i) contracted_sig_blocks[5][i] = contracted_seq05_arr[i];
    for(size_t i = 0; i < 2; ++i) contracted_sig_blocks[6][i] = contracted_seq06_arr[i];
    for(size_t i = 0; i < 2; ++i) contracted_sig_blocks[7][i] = contracted_seq07_arr[i];
    for(size_t i = 0; i < 2; ++i) contracted_sig_blocks[8][i] = contracted_seq08_arr[i];
    for(size_t i = 0; i < 2; ++i) contracted_sig_blocks[9][i] = contracted_seq09_arr[i];
    for(size_t i = 0; i < 2; ++i) contracted_sig_blocks[10][i] = contracted_seq10_arr[i];

    sparse_bispace<2> two_d_correct = spb_1 % spb_2 << contracted_sig_blocks; 
    if(two_d != two_d_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::contract(...) returned incorrect value");
    }
}

//Does the contracted bispace have the correct nnz
void sparse_bispace_test::test_contract_3d_sparse_2_nnz() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_contract_3d_sparse_2_nnz()";

    //Bispaces
    sparse_bispace<1> spb_1(8);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(9);
    std::vector<size_t> split_points_2; 
    split_points_2.push_back(3);
    split_points_2.push_back(6);
    split_points_2.push_back(8);
    spb_2.split(split_points_2);

    sparse_bispace<1> spb_3(10);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(4);
    split_points_3.push_back(7);
    spb_3.split(split_points_3);

    //Sparsity Info
    size_t seq00_arr[3] = {0,0,0};
    size_t seq01_arr[3] = {0,0,2};
    size_t seq02_arr[3] = {0,1,2};
    size_t seq03_arr[3] = {0,2,1};
    size_t seq04_arr[3] = {0,2,2};
    size_t seq05_arr[3] = {0,3,1};
    size_t seq06_arr[3] = {1,0,1};
    size_t seq07_arr[3] = {1,0,2};
    size_t seq08_arr[3] = {1,1,0};
    size_t seq09_arr[3] = {1,2,0};
    size_t seq10_arr[3] = {1,3,1};
    size_t seq11_arr[3] = {2,1,1};
    size_t seq12_arr[3] = {2,2,2};
    size_t seq13_arr[3] = {2,3,0};

    std::vector< sequence<3,size_t> > sig_blocks(14);
    for(size_t i = 0; i < 3; ++i) sig_blocks[0][i] = seq00_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[1][i] = seq01_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[2][i] = seq02_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[3][i] = seq03_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[4][i] = seq04_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[5][i] = seq05_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[6][i] = seq06_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[7][i] = seq07_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[8][i] = seq08_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[9][i] = seq09_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[10][i] = seq10_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[11][i] = seq11_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[12][i] = seq12_arr[i];
    for(size_t i = 0; i < 3; ++i) sig_blocks[13][i] = seq13_arr[i];

    sparse_bispace<3> three_d = spb_1 % spb_2 % spb_3 << sig_blocks;
    sparse_bispace<2> two_d = three_d.contract(2); 

    if(two_d.get_nnz() != 63)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::contract(...) returned incorrect value");
    }
}

//In this test, we get rid of all the sparsity in a bispace by calling contract()
void sparse_bispace_test::test_contract_3d_sparse_destroy_all_sparsity() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_contract_3d_sparse_destroy_all_sparsity()";

    //Bispaces
    sparse_bispace<1> spb_1(8);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    spb_1.split(split_points_1);

    sparse_bispace<1> spb_2(9);
    std::vector<size_t> split_points_2; 
    split_points_2.push_back(3);
    split_points_2.push_back(6);
    split_points_2.push_back(8);
    spb_2.split(split_points_2);

    sparse_bispace<1> spb_3(10);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(4);
    split_points_3.push_back(7);
    spb_3.split(split_points_3);

    //Sparsity
    std::vector< sequence<2,size_t> > sig_blocks(5); 
    sig_blocks[0][0] = 0;
    sig_blocks[0][1] = 1;
    sig_blocks[1][0] = 0;
    sig_blocks[1][1] = 3;
    sig_blocks[2][0] = 1;
    sig_blocks[2][1] = 0;
    sig_blocks[3][0] = 1;
    sig_blocks[3][1] = 2;
    sig_blocks[4][0] = 2;
    sig_blocks[4][1] = 1;

    sparse_bispace<3> three_d = spb_1 % spb_2 << sig_blocks | spb_3;
    sparse_bispace<2> two_d = three_d.contract(1); 


    sparse_bispace<2> two_d_correct = spb_1 | spb_3; 
    if(two_d != two_d_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::contract(...) returned incorrect value");
    }
}

void sparse_bispace_test::test_fuse_2d_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_fuse_2d_2d()";

    //Sparsity data 1
    size_t seq0_arr[2] = {1,2};
    size_t seq1_arr[2] = {1,5};
    size_t seq2_arr[2] = {2,3};
    size_t seq3_arr[2] = {4,1};
    size_t seq4_arr[2] = {4,4};
    size_t seq5_arr[2] = {5,1};
    size_t seq6_arr[2] = {5,2};

    std::vector< sequence<2,size_t> > sig_blocks_1(7); 
    for(size_t i = 0; i < 2; ++i) sig_blocks_1[0][i] = seq0_arr[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_1[1][i] = seq1_arr[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_1[2][i] = seq2_arr[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_1[3][i] = seq3_arr[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_1[4][i] = seq4_arr[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_1[5][i] = seq5_arr[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_1[6][i] = seq6_arr[i];

    //Need 6 blocks
    sparse_bispace<1> spb_i(12);
    std::vector<size_t> split_points_i;
    split_points_i.push_back(2);
    split_points_i.push_back(4);
    split_points_i.push_back(6);
    split_points_i.push_back(9);
    split_points_i.push_back(11);
    spb_i.split(split_points_i);

    //Need 6 blocks
    sparse_bispace<1> spb_k(17); 
    std::vector<size_t> split_points_k;
    split_points_k.push_back(3);
    split_points_k.push_back(7);
    split_points_k.push_back(10);
    split_points_k.push_back(12);
    split_points_k.push_back(14);
    spb_k.split(split_points_k);

    sparse_bispace<2> spb_A = spb_i % spb_k << sig_blocks_1;

    //Sparsity data 2
    size_t seq0_arr_2[2] = {1,2};
    size_t seq1_arr_2[2] = {1,6};
    size_t seq2_arr_2[2] = {2,5};
    size_t seq3_arr_2[2] = {2,9};
    size_t seq4_arr_2[2] = {3,1};
    size_t seq5_arr_2[2] = {4,3};
    size_t seq6_arr_2[2] = {4,6};
    size_t seq7_arr_2[2] = {5,1};
    size_t seq8_arr_2[2] = {5,5};

    std::vector< sequence<2,size_t> > sig_blocks_2(9); 
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[0][i] = seq0_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[1][i] = seq1_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[2][i] = seq2_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[3][i] = seq3_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[4][i] = seq4_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[5][i] = seq5_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[6][i] = seq6_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[7][i] = seq7_arr_2[i];
    for(size_t i = 0; i < 2; ++i) sig_blocks_2[8][i] = seq8_arr_2[i];

    //Need 10 blocks
    sparse_bispace<1> spb_j(25);
    std::vector<size_t> split_points_j;
    split_points_j.push_back(4);
    split_points_j.push_back(7);
    split_points_j.push_back(9);
    split_points_j.push_back(12);
    split_points_j.push_back(15);
    split_points_j.push_back(17);
    split_points_j.push_back(18);
    split_points_j.push_back(21);
    split_points_j.push_back(23);
    spb_j.split(split_points_j);

    sparse_bispace<2> spb_B = spb_k % spb_j << sig_blocks_2;

    sparse_bispace<3> spb_C = spb_A.fuse(spb_B);

    //Correct bispace 
    size_t correct_seq00_arr[3] = {1,2,5};
    size_t correct_seq01_arr[3] = {1,2,9};
    size_t correct_seq02_arr[3] = {1,5,1};
    size_t correct_seq03_arr[3] = {1,5,5};
    size_t correct_seq04_arr[3] = {2,3,1};
    size_t correct_seq05_arr[3] = {4,1,2};
    size_t correct_seq06_arr[3] = {4,1,6};
    size_t correct_seq07_arr[3] = {4,4,3};
    size_t correct_seq08_arr[3] = {4,4,6};
    size_t correct_seq09_arr[3] = {5,1,2};
    size_t correct_seq10_arr[3] = {5,1,6};
    size_t correct_seq11_arr[3] = {5,2,5};
    size_t correct_seq12_arr[3] = {5,2,9};

    std::vector< sequence<3,size_t> > correct_sig_blocks(13);
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[0][i] = correct_seq00_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[1][i] = correct_seq01_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[2][i] = correct_seq02_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[3][i] = correct_seq03_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[4][i] = correct_seq04_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[5][i] = correct_seq05_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[6][i] = correct_seq06_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[7][i] = correct_seq07_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[8][i] = correct_seq08_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[9][i] = correct_seq09_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[10][i] = correct_seq10_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[11][i] = correct_seq11_arr[i];
    for(size_t i = 0; i < 3; ++i) correct_sig_blocks[12][i] = correct_seq12_arr[i];

    sparse_bispace<3> spb_C_correct = spb_i % spb_k % spb_j << correct_sig_blocks;

    if(spb_C != spb_C_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::fuse(...) returned incorrect value");
    }
}

//Tests that fuse gives the right answer when there is no overlap of sparsity
void sparse_bispace_test::test_fuse_3d_3d_no_overlap() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_fuse_3d_3d_no_overlap()";

    sparse_bispace<1> spb_1(11);
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    split_points_1.push_back(9);
    spb_1.split(split_points_1);

    std::vector< sequence<2,size_t> > sig_blocks(4);
    sig_blocks[0][0] = 0; 
    sig_blocks[0][1] = 1;
    sig_blocks[1][0] = 1;
    sig_blocks[1][1] = 2;
    sig_blocks[2][0] = 2;
    sig_blocks[2][1] = 3;
    sig_blocks[3][0] = 3;
    sig_blocks[3][1] = 2;

    sparse_bispace<1> spb_2(9);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    split_points_2.push_back(5);
    spb_2.split(split_points_2);

    sparse_bispace<2> two_d_1 = spb_1 % spb_1 << sig_blocks;
    sparse_bispace<3> three_d_1 =  two_d_1 | spb_2;
    sparse_bispace<3> three_d_2 = spb_2 | two_d_1;
    sparse_bispace<5> five_d_1 = three_d_1.fuse(three_d_2);
    sparse_bispace<5> five_d_correct = two_d_1 | spb_2 | two_d_1;

    if(five_d_1 != five_d_correct)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::fuse(...) returned incorrect value");
    }
}

//Should throw exception - can't fuse if fuse point doesn't match
void sparse_bispace_test::test_fuse_3d_3d_invalid_no_match() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_fuse_3d_3d_invalid_no_match()";

    sparse_bispace<1> spb_1(11);
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    split_points_1.push_back(9);
    spb_1.split(split_points_1);

    std::vector< sequence<2,size_t> > sig_blocks(4);
    sig_blocks[0][0] = 0; 
    sig_blocks[0][1] = 1;
    sig_blocks[1][0] = 1;
    sig_blocks[1][1] = 2;
    sig_blocks[2][0] = 2;
    sig_blocks[2][1] = 3;
    sig_blocks[3][0] = 3;
    sig_blocks[3][1] = 2;

    sparse_bispace<1> spb_2(9);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    split_points_2.push_back(5);
    spb_2.split(split_points_2);

    //So as not to match spb_2
    sparse_bispace<1> spb_3(9);
    std::vector<size_t> split_points_3;
    split_points_3.push_back(2);
    spb_3.split(split_points_3);

    sparse_bispace<2> two_d_1 = spb_1 % spb_1 << sig_blocks;
    sparse_bispace<3> three_d_1 =  two_d_1 | spb_2;
    sparse_bispace<3> three_d_2 = spb_3 | two_d_1;

    bool threw_exception = false;
    try
    {
        sparse_bispace<5> five_d_1 = three_d_1.fuse(three_d_2);
    }
    catch(bad_parameter&)
    {
        threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::fuse(...) did not throw exception for incompatible bispaces");
    }
}

void sparse_bispace_test::test_get_n_index_groups() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_n_index_groups()";
    
    index_groups_test_f tf = index_groups_test_f();

    if(tf.bispace.get_n_index_groups() != 5)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::get_n_index_groups(...) did not return correct value");
    }
}

void sparse_bispace_test::test_get_index_group_offset() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_index_group_offset()";
    
    index_groups_test_f tf = index_groups_test_f();

    size_t correct_offsets[5] = {0,1,2,4,5};
    for(size_t grp = 0; grp < tf.bispace.get_n_index_groups(); ++grp)
    {
        if(tf.bispace.get_index_group_offset(grp) != correct_offsets[grp])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_bispace<N>::get_index_group_offset(...) did not return correct value");
        }
    }
}

void sparse_bispace_test::test_get_index_group_order() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_index_group_order()";
    
    index_groups_test_f tf = index_groups_test_f();

    size_t correct_orders[5] = {1,1,2,1,2};
    for(size_t grp = 0; grp < tf.bispace.get_n_index_groups(); ++grp)
    {
        if(tf.bispace.get_index_group_order(grp) != correct_orders[grp])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_bispace<N>::get_index_group_order(...) did not return correct value");
        }
    }
}

void sparse_bispace_test::test_get_index_group_dim() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_index_group_dim()";
    
    index_groups_test_f tf = index_groups_test_f();

    size_t correct_dims[5] = {10,10,24,10,27};
    for(size_t grp = 0; grp < tf.bispace.get_n_index_groups(); ++grp)
    {
        if(tf.bispace.get_index_group_dim(grp) != correct_dims[grp])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_bispace<N>::get_index_group_dim(...) did not return correct value");
        }
    }
}

void sparse_bispace_test::test_get_index_group_containing_subspace() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_index_group_containing_subspace()";
    
    index_groups_test_f tf = index_groups_test_f();

    size_t correct_index_groups[3] = {1,2,4};
    size_t subspaces[3] = {1,3,6};
    for(size_t subspace_idx = 0; subspace_idx < 3; ++subspace_idx)
    {
        if(tf.bispace.get_index_group_containing_subspace(subspaces[subspace_idx]) != correct_index_groups[subspace_idx])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_bispace<N>::get_index_group_containing_subspace(...) did not return correct value");
        }
    }
}

//Get the correct number of elements for a 2d tensor with both indices coupled by sparsity
void sparse_bispace_test::test_get_nnz_2d_sparsity() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_nnz_2d_sparsity()";

    sparse_bispace<1> spb_1(11);

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     * 3: 2
     */
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    split_points_1.push_back(9);
    spb_1.split(split_points_1);

    std::vector< sequence<2,size_t> > sig_blocks(4);
    sig_blocks[0][0] = 0; 
    sig_blocks[0][1] = 1;//block size 6
    sig_blocks[1][0] = 1;
    sig_blocks[1][1] = 2;//block size 12 
    sig_blocks[2][0] = 2;
    sig_blocks[2][1] = 3;//block size 8 
    sig_blocks[3][0] = 3;
    sig_blocks[3][1] = 2;//block size 8

    sparse_bispace<2> two_d = spb_1 % spb_1 << sig_blocks; 

    if(two_d.get_nnz() != 34)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::get_nnz(...) returned incorrect value");
    }
}

//Get the correct number of elements for a 3d tensor with the second two indices coupled by sparsity 
void sparse_bispace_test::test_get_nnz_3d_dense_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_nnz_3d_dense_sparse()";

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     * 3: 2
     */
    sparse_bispace<1> spb_1(11);
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    split_points_1.push_back(9);
    spb_1.split(split_points_1);

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     */
    sparse_bispace<1> spb_2(9);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    split_points_2.push_back(5);
    spb_2.split(split_points_2);


    //Total sparse nnz 34 
    std::vector< sequence<2,size_t> > sig_blocks(4);
    sig_blocks[0][0] = 0; 
    sig_blocks[0][1] = 1;//block size 6
    sig_blocks[1][0] = 1;
    sig_blocks[1][1] = 2;//block size 12 
    sig_blocks[2][0] = 2;
    sig_blocks[2][1] = 3;//block size 8 
    sig_blocks[3][0] = 3;
    sig_blocks[3][1] = 2;//block size 8

    sparse_bispace<3> three_d = spb_2 | spb_1 % spb_1 << sig_blocks; 

    if(three_d.get_nnz() != 306)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::get_nnz(...) returned incorrect value");
    }
}

//Get the correct number of elements for a 3d tensor with the second two indices coupled by sparsity 
void sparse_bispace_test::test_get_nnz_3d_sparse_dense() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_nnz_3d_sparse_dense()";

    sparse_bispace<1> spb_1(11);

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     * 3: 2
     */
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    split_points_1.push_back(9);
    spb_1.split(split_points_1);

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     */
    sparse_bispace<1> spb_2(9);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    split_points_2.push_back(5);
    spb_2.split(split_points_2);


    //Total sparse nnz 34 
    std::vector< sequence<2,size_t> > sig_blocks(4);
    sig_blocks[0][0] = 0; 
    sig_blocks[0][1] = 1;//block size 6
    sig_blocks[1][0] = 1;
    sig_blocks[1][1] = 2;//block size 12 
    sig_blocks[2][0] = 2;
    sig_blocks[2][1] = 3;//block size 8 
    sig_blocks[3][0] = 3;
    sig_blocks[3][1] = 2;//block size 8

    sparse_bispace<3> three_d = spb_1 % spb_1 << sig_blocks | spb_2;

    if(three_d.get_nnz() != 306)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::get_nnz(...) returned incorrect value");
    }
}

//Get the correct number of elements for a 3d tensor with all indices coupled by sparsity 
void sparse_bispace_test::test_get_nnz_3d_fully_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_nnz_3d_fully_sparse()";

    sparse_bispace<1> spb_1(11);

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     * 3: 2
     */
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    split_points_1.push_back(9);
    spb_1.split(split_points_1);

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     */
    sparse_bispace<1> spb_2(9);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    split_points_2.push_back(5);
    spb_2.split(split_points_2);


    //Total sparse nnz 34 
    std::vector< sequence<3,size_t> > sig_blocks(5);

    sig_blocks[0][0] = 0; 
    sig_blocks[0][1] = 0;
    sig_blocks[0][2] = 2;//block size 16
    sig_blocks[1][0] = 0; 
    sig_blocks[1][1] = 0;
    sig_blocks[1][2] = 3;//block size 8
    sig_blocks[2][0] = 1;
    sig_blocks[2][1] = 2;
    sig_blocks[2][2] = 2;//block size 48
    sig_blocks[3][0] = 1;
    sig_blocks[3][1] = 3;
    sig_blocks[3][2] = 1;//block size 18
    sig_blocks[4][0] = 2;
    sig_blocks[4][1] = 0;
    sig_blocks[4][2] = 1;//block size 24
    //TOTAL 78

    sparse_bispace<3> three_d = spb_2 % spb_1 % spb_1 << sig_blocks; 

    if(three_d.get_nnz() != 114)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::get_nnz(...) returned incorrect value");
    }
}

void sparse_bispace_test::test_equality_false_sparsity_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_equality_false_diff_sparsity_2d()";

    sparse_bispace<1> spb_1(11);

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     * 3: 2
     */
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    split_points_1.push_back(9);
    spb_1.split(split_points_1);

    //Specify different sets of significant blocks
    std::vector< sequence<2,size_t> > sig_blocks_1(4);
    sig_blocks_1[0][0] = 0; 
    sig_blocks_1[0][1] = 1;
    sig_blocks_1[1][0] = 1;
    sig_blocks_1[1][1] = 2;
    sig_blocks_1[2][0] = 2;
    sig_blocks_1[2][1] = 3;
    sig_blocks_1[3][0] = 3;
    sig_blocks_1[3][1] = 2;

    std::vector< sequence<2,size_t> > sig_blocks_2(4);
    sig_blocks_2[0][0] = 0; 
    sig_blocks_2[0][1] = 1;
    sig_blocks_2[1][0] = 1;
    sig_blocks_2[1][1] = 2;
    sig_blocks_2[2][0] = 2;
    sig_blocks_2[2][1] = 1;//! Changed this one value
    sig_blocks_2[3][0] = 3;
    sig_blocks_2[3][1] = 2;

    sparse_bispace<2> two_d_1 = spb_1 % spb_1 << sig_blocks_1; 
    sparse_bispace<2> two_d_2 = spb_1 % spb_1 << sig_blocks_2; 

    if(two_d_1 == two_d_2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::operator==(...) returned incorrect value");
    }
}

void sparse_bispace_test::test_equality_true_sparsity_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_equality_true_sparsity_2d()";

    sparse_bispace<1> spb_1(11);

    /* Splitting pattern results in the following block sizes:
     * 0: 2
     * 1: 3
     * 2: 4
     * 3: 2
     */
    std::vector<size_t> split_points_1; 
    split_points_1.push_back(2);
    split_points_1.push_back(5);
    split_points_1.push_back(9);
    spb_1.split(split_points_1);

    //Specify different sets of significant blocks
    std::vector< sequence<2,size_t> > sig_blocks_1(4);
    sig_blocks_1[0][0] = 0; 
    sig_blocks_1[0][1] = 1;
    sig_blocks_1[1][0] = 1;
    sig_blocks_1[1][1] = 2;
    sig_blocks_1[2][0] = 2;
    sig_blocks_1[2][1] = 3;
    sig_blocks_1[3][0] = 3;
    sig_blocks_1[3][1] = 2;

    sparse_bispace<2> two_d_1 = spb_1 % spb_1 << sig_blocks_1; 
    sparse_bispace<2> two_d_2 = spb_1 % spb_1 << sig_blocks_1; 

    if(two_d_1 != two_d_2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::operator==(...) returned incorrect value");
    }
}

} // namespace libtensor
