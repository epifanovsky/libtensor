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

} // namespace libtensor
