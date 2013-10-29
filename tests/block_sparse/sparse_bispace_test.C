#include <libtensor/block_sparse/sparse_bispace.h>
#include <libtensor/core/out_of_bounds.h>
#include <vector>
#include "sparse_bispace_test.h"

//TODO: REMOVE
#include <iostream>

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

        test_get_block_offset_1d(); 
        test_get_block_offset_1d_empty_vec();
        test_get_block_offset_1d_oob();
        test_get_block_offset_2d(); 

        test_get_block_offset_canonical_1d(); 
        test_get_block_offset_canonical_1d_empty_vec();
        test_get_block_offset_canonical_1d_oob();
        test_get_block_offset_canonical_2d(); 

        test_get_nnz_2d_sparsity();
        test_get_nnz_3d_dense_sparse();
        test_get_nnz_3d_sparse_dense();
        test_get_nnz_3d_fully_sparse();
        test_get_block_offset_2d_sparsity();
        test_get_block_offset_3d_dense_sparse();
        test_get_block_offset_3d_sparse_dense();
        test_get_block_offset_3d_fully_sparse();
}

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


void sparse_bispace_test::test_get_block_offset_1d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_block_offset_1d()";
    sparse_bispace<1> spb(5);
    std::vector<size_t> split_points;
    split_points.push_back(2);
    split_points.push_back(3);
    spb.split(split_points);

    std::vector<size_t> tile_indices(1,1);
    if(spb.get_block_offset(tile_indices) != 2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::get_block_offset(...) returned incorrect value");

    }
}

void sparse_bispace_test::test_get_block_offset_1d_empty_vec() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_block_offset_1d_empty_vec()";
    sparse_bispace<1> spb(5);
    std::vector<size_t> split_points;
    split_points.push_back(2);
    split_points.push_back(3);
    spb.split(split_points);

    std::vector<size_t> tile_indices;
    bool threw_exception = false;
    try
    {
        spb.get_block_offset(tile_indices);
    }
    catch(out_of_bounds&)
    {
        threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::get_block_offset(...) did not throw exception when passed empty vector");

    }
}

void sparse_bispace_test::test_get_block_offset_1d_oob() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_block_offset_1d_oob()";
    sparse_bispace<1> spb(5);
    std::vector<size_t> split_points;
    split_points.push_back(2);
    split_points.push_back(3);
    spb.split(split_points);

    std::vector<size_t> tile_indices(1,3);
    bool threw_exception = false;
    try
    {
        spb.get_block_offset(tile_indices);
    }
    catch(out_of_bounds&)
    {
        threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::get_block_offset(...) did not throw exception when passed invalid tile indices");

    }
}

void sparse_bispace_test::test_get_block_offset_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_block_offset_2d()";

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

    std::vector<size_t> tile_indices;
    tile_indices.push_back(2); //1d size is 2
    tile_indices.push_back(1); //1d size is 3

    if(two_d.get_block_offset(tile_indices) != 22)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::get_block_offset(...) returned incorrect value");
    }
}

void sparse_bispace_test::test_get_block_offset_canonical_1d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_block_offset_canonical_1d()";
    sparse_bispace<1> spb(5);
    std::vector<size_t> split_points;
    split_points.push_back(2);
    split_points.push_back(3);
    spb.split(split_points);

    std::vector<size_t> tile_indices(1,1);
    if(spb.get_block_offset_canonical(tile_indices) != 2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::get_block_offset_canonical(...) returned incorrect value");

    }
}

void sparse_bispace_test::test_get_block_offset_canonical_1d_empty_vec() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_block_offset_canonical_1d_empty_vec()";
    sparse_bispace<1> spb(5);
    std::vector<size_t> split_points;
    split_points.push_back(2);
    split_points.push_back(3);
    spb.split(split_points);

    std::vector<size_t> tile_indices;
    bool threw_exception = false;
    try
    {
        spb.get_block_offset_canonical(tile_indices);
    }
    catch(out_of_bounds&)
    {
        threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::get_block_offset_canonical(...) did not throw exception when passed empty vector");

    }
}

void sparse_bispace_test::test_get_block_offset_canonical_1d_oob() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_block_offset_canonical_1d_oob()";
    sparse_bispace<1> spb(5);
    std::vector<size_t> split_points;
    split_points.push_back(2);
    split_points.push_back(3);
    spb.split(split_points);

    std::vector<size_t> tile_indices(1,3);
    bool threw_exception = false;
    try
    {
        spb.get_block_offset_canonical(tile_indices);
    }
    catch(out_of_bounds&)
    {
        threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::get_block_offset_canonical(...) did not throw exception when passed invalid tile indices");

    }
}

void sparse_bispace_test::test_get_block_offset_canonical_2d() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_block_offset_canonical_2d()";

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

    std::vector<size_t> tile_indices;
    tile_indices.push_back(2); //Starts at 3
    tile_indices.push_back(1); //Starts at 2

    if(two_d.get_block_offset_canonical(tile_indices) != 20)
    {
        std::cout << two_d.get_block_offset_canonical(tile_indices) << "\n";
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_bispace<N>::get_block_offset_canonical(...) returned incorrect value");

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

//Get the correct offset of each block in a 2d tensor with both indices coupled by sparsity 
void sparse_bispace_test::test_get_block_offset_2d_sparsity() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_block_offset_2d_sparsity()";

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

    std::vector<size_t> correct_offsets;
    correct_offsets.push_back(0);
    correct_offsets.push_back(6);
    correct_offsets.push_back(18);
    correct_offsets.push_back(26);

    for(size_t i = 0; i < sig_blocks.size(); ++i)
    {
        std::vector<size_t> block_key(2);
        block_key[0] = sig_blocks[i][0];
        block_key[1] = sig_blocks[i][1];
        if(two_d.get_block_offset(block_key) != correct_offsets[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_bispace<N>::get_block_offset(...) returned incorrect value");

        }
    }
}

void sparse_bispace_test::test_get_block_offset_3d_dense_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_block_offset_3d_dense_sparse()";

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

    sparse_bispace<3> three_d = spb_2 | spb_1 % spb_1 << sig_blocks; 

    //Three arbitrarily chosen block vectors
    std::vector< std::vector<size_t> > block_keys(3);
    block_keys[0].push_back(0);
    block_keys[0].push_back(1);
    block_keys[0].push_back(2); //offset = 0*34+2*6
    block_keys[1].push_back(1);
    block_keys[1].push_back(0);
    block_keys[1].push_back(1); //offset = 2*34 + 0 = 68
    block_keys[2].push_back(2);
    block_keys[2].push_back(2);
    block_keys[2].push_back(3); //offset = 5*34 + 4*18 = 242

    std::vector<size_t> correct_offsets;
    correct_offsets.push_back(12);
    correct_offsets.push_back(68);
    correct_offsets.push_back(242);

    for(size_t i = 0; i < block_keys.size(); ++i)
    {
        if(three_d.get_block_offset(block_keys[i]) != correct_offsets[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_bispace<N>::get_block_offset(...) returned incorrect value");

        }
    }
}

void sparse_bispace_test::test_get_block_offset_3d_sparse_dense() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_block_offset_3d_sparse_dense()";

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

    //Three arbitrarily chosen block vectors
    std::vector< std::vector<size_t> > block_keys(3);
    block_keys[0].push_back(0);
    block_keys[0].push_back(1);
    block_keys[0].push_back(1); //offset = 0*9 + 6*2 = 12
    block_keys[1].push_back(1);
    block_keys[1].push_back(2);
    block_keys[1].push_back(0); //offset = 6*9 + 12*0 = 54
    block_keys[2].push_back(2);
    block_keys[2].push_back(3);
    block_keys[2].push_back(2); //offset = 18*9 + 8*5 = 202

    std::vector<size_t> correct_offsets;
    correct_offsets.push_back(12);
    correct_offsets.push_back(54);
    correct_offsets.push_back(202);

    for(size_t i = 0; i < block_keys.size(); ++i)
    {
        if(three_d.get_block_offset(block_keys[i]) != correct_offsets[i])
        {
            std::cout << "\ni: " << i << "\n";
            std::cout << "mine: " << three_d.get_block_offset(block_keys[i])  << "\n";
            std::cout << "correct: " << correct_offsets[i] << "\n";
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_bispace<N>::get_block_offset(...) returned incorrect value");

            exit(1);
        }
    }
}

void sparse_bispace_test::test_get_block_offset_3d_fully_sparse() throw(libtest::test_exception)
{
    static const char *test_name = "sparse_bispace_test::test_get_block_offset_3d_fully_sparse()";

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


    //Total sparse block_offset 34 
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

    std::vector<size_t> correct_offsets;
    correct_offsets.push_back(0);
    correct_offsets.push_back(16);
    correct_offsets.push_back(24);
    correct_offsets.push_back(72);
    correct_offsets.push_back(90);
    
    for(size_t i = 0; i < sig_blocks.size(); ++i)
    {
        std::vector<size_t> block_key(3);
        block_key[0] = sig_blocks[i][0];
        block_key[1] = sig_blocks[i][1];
        block_key[2] = sig_blocks[i][2];
        if(three_d.get_block_offset(block_key) != correct_offsets[i])
        {
            fail_test(test_name,__FILE__,__LINE__,
                    "sparse_bispace<N>::get_block_offset(...) returned incorrect value");
        }
    }
}

} // namespace libtensor
