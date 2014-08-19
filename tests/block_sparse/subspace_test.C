#include "subspace_test.h"
#include <libtensor/block_sparse/subspace.h>

namespace libtensor {

void subspace_test::perform() throw(libtest::test_exception)
{
    test_get_dim();

    test_split_not_strictly_increasing();
    test_split_not_strictly_increasing_two_calls();
    test_split_gt_upper_bound();
    test_split_zero_size();

    test_equality_true();
    test_equality_false_diff_dims();
    test_equality_false_diff_splits();

    test_get_block_abs_index_one_block();
    test_get_block_abs_index_two_block();
    test_get_block_abs_index_gt_upper_bound();
}

void subspace_test::test_get_dim() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_test::test_get_dim()";
    subspace spb(5);
    if(spb.get_dim() != 5)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace::get_dim(...) returned incorrect value");
    } 
}

void subspace_test::test_get_n_blocks() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_test::test_get_n_blocks()";
    subspace spb(5);
    std::vector<size_t> split_points; 
    split_points.push_back(1);
    split_points.push_back(3);
    spb.split(split_points);

    if(spb.get_n_blocks() != 3)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace::get_n_blocks(...) returned incorrect value");
    } 
}



/* Should throw out_of_bounds when split points are passed with an order
 * that is not strictly increasing.
 *
 */
void subspace_test::test_split_not_strictly_increasing() throw(libtest::test_exception) {

    static const char *test_name = "subspace_test::test_split_not_strictly_increasing()";

    bool threw_exception = false;
    subspace spb(5);
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
                "subspace::split(...) did not throw exception when split points not strictly increasing");
    }
}

/* Should throw out_of_bounds when split points are passed with an order
 * that is not strictly increasing, over two consecutive calls
 *
 */
void subspace_test::test_split_not_strictly_increasing_two_calls() throw(libtest::test_exception) {

    static const char *test_name = "subspace_test::test_split_not_strictly_increasing_two_calls()";

    bool threw_exception = false;
    subspace spb(5);
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
                "subspace::split(...) did not throw exception when split points not strictly increasing");
    }
}

/* Should throw out_of_bounds when a split_point that exceeds the dimension of the space is passed 
 */
void subspace_test::test_split_gt_upper_bound() throw(libtest::test_exception) {

    static const char *test_name = "subspace_test::test_split_gt_upper_bound()";

    bool threw_exception = false;
    subspace spb(5);
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
                "subspace::split(...) did not throw exception when split point index greater than max index value");
    }
}

/* Should throw out_of_bounds when zero split_points are specified
 */
void subspace_test::test_split_zero_size() throw(libtest::test_exception) {

    static const char *test_name = "subspace_test::test_split_zero_size()";

    bool threw_exception = false;
    subspace spb(5);
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
                "subspace::split(...) did not throw exception when zero points specified");
    }
}

void subspace_test::test_equality_true() throw(libtest::test_exception) { 

    static const char *test_name = "subspace_test::test_equality_true()";

    subspace spb_1(5);
    subspace spb_2(5);
    std::vector<size_t> split_points;
    split_points.push_back(3);
    split_points.push_back(4);
    spb_1.split(split_points); 
    spb_2.split(split_points); 

    if(!(spb_1 == spb_2)) 
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace::operator==(...) returned false");

    }
}

void subspace_test::test_equality_false_diff_dims() throw(libtest::test_exception) {

    static const char *test_name = "subspace_test::test_equality_false_diff_dims()";

    subspace spb_1(5);
    subspace spb_2(6);
    std::vector<size_t> split_points;
    split_points.push_back(3);
    split_points.push_back(4);
    spb_1.split(split_points); 
    spb_2.split(split_points); 

    if(spb_1 == spb_2) 
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace::operator==(...) returned true");
    }
}

void subspace_test::test_equality_false_diff_splits() throw(libtest::test_exception) {

    static const char *test_name = "subspace_test::test_equality_false_diff_dims()";

    subspace spb_1(5);
    subspace spb_2(5);
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
                "subspace::operator==(...) returned true");
    }
}

/* Should return zero
 */
void subspace_test::test_get_block_abs_index_one_block() throw(libtest::test_exception) {
    static const char *test_name = "subspace_test::test_get_block_abs_index_one_block()";
    subspace spb(5);
    if(! (spb.get_block_abs_index(0) == 0))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace::get_block_abs_index(0) did not return zero");
    }
}

/* Should return 4
 */
void subspace_test::test_get_block_abs_index_two_block() throw(libtest::test_exception) {
    static const char *test_name = "subspace_test::test_get_block_abs_index_one_block()";
    subspace spb(5);
    std::vector<size_t> split_points;
    split_points.push_back(4);
    spb.split(split_points);
    if(! (spb.get_block_abs_index(1) == 4))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace::get_block_abs_index(1) did not return 4");
    }
}

 /* Should throw out_of_bounds
  */ 
void subspace_test::test_get_block_abs_index_gt_upper_bound() throw(libtest::test_exception) {
    static const char *test_name = "subspace_test::test_get_block_abs_index_gt_upper_bound()";
    bool threw_exception = false;
    subspace spb(5);
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
                "subspace::get_block_abs_index(...) did not throw exception when when block_idx > (# of blocks - 1)");
    }
}

} // namespace libtensor
