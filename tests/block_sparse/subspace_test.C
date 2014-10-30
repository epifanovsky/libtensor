#include "subspace_test.h"
#include <libtensor/block_sparse/subspace.h>

namespace libtensor {

void subspace_test::perform() throw(libtest::test_exception)
{
#if 0
    test_get_dim();
    test_get_n_blocks();

    test_split_zero_first();
    test_split_resplit();
    test_split_not_strictly_increasing();
    test_split_gt_upper_bound();
    test_split_zero_size();

    test_equality_true();
    test_equality_false_diff_dims();
    test_equality_false_diff_splits();

    test_get_block_abs_index_one_block();
    test_get_block_abs_index_two_block();
    test_get_block_abs_index_gt_upper_bound();
#endif
}

#if 0
void subspace_test::test_get_dim() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_test::test_get_dim()";
    subspace sub(5);
    if(sub.get_dim() != 5)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace::get_dim(...) returned incorrect value");
    } 
}

void subspace_test::test_get_n_blocks() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_test::test_get_n_blocks()";
    subspace sub(5);
    std::vector<size_t> split_points; 
    split_points.push_back(1);
    split_points.push_back(3);
    sub.split(split_points);

    if(sub.get_n_blocks() != 3)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace::get_n_blocks(...) returned incorrect value");
    } 
}

void subspace_test::test_split_zero_first() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_test::test_split_zero_first()";

    subspace sub(5);
    sub.split(idx_list(1,0));
    if(sub.get_n_blocks() != 1 || 
       sub.get_block_size(0) != 5 || 
       sub.get_block_abs_index(0) != 0)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace::split(...) performed incorrectly");
    }
}

void subspace_test::test_split_resplit() throw(libtest::test_exception)
{
    static const char *test_name = "subspace_test::test_split_resplit()";

    idx_list split_points(1,2);
    split_points.push_back(3);
    subspace sub(5,split_points);
    if(sub.get_n_blocks() != 3 || 
       sub.get_block_size(0) != 2 || 
       sub.get_block_size(1) != 1 || 
       sub.get_block_size(2) != 2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace::split(...) did not produce correct result");
    }

    sub.split(idx_list(1,0));
    if(sub.get_n_blocks() != 1 || 
       sub.get_block_size(0) != 5 || 
       sub.get_block_abs_index(0) != 0)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace::split(...) did not produce correct result");
    }
}



/* Should throw out_of_bounds when split points are passed with an order
 * that is not strictly increasing.
 *
 */
void subspace_test::test_split_not_strictly_increasing() throw(libtest::test_exception) {

    static const char *test_name = "subspace_test::test_split_not_strictly_increasing()";

    bool threw_exception = false;
    subspace sub(5);
    std::vector<size_t> split_points;
    split_points.push_back(3);
    split_points.push_back(1);
    try
    {
        sub.split(split_points);
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
    subspace sub(5);
    std::vector<size_t> split_points;
    split_points.push_back(5);
    try
    {
        sub.split(split_points);
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
    subspace sub(5);
    std::vector<size_t> split_points;
    try
    {
        sub.split(split_points);
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

    subspace sub_1(5);
    subspace sub_2(5);
    std::vector<size_t> split_points;
    split_points.push_back(3);
    split_points.push_back(4);
    sub_1.split(split_points); 
    sub_2.split(split_points); 

    if(!(sub_1 == sub_2)) 
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace::operator==(...) returned false");

    }
}

void subspace_test::test_equality_false_diff_dims() throw(libtest::test_exception) {

    static const char *test_name = "subspace_test::test_equality_false_diff_dims()";

    subspace sub_1(5);
    subspace sub_2(6);
    std::vector<size_t> split_points;
    split_points.push_back(3);
    split_points.push_back(4);
    sub_1.split(split_points); 
    sub_2.split(split_points); 

    if(sub_1 == sub_2) 
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace::operator==(...) returned true");
    }
}

void subspace_test::test_equality_false_diff_splits() throw(libtest::test_exception) {

    static const char *test_name = "subspace_test::test_equality_false_diff_dims()";

    subspace sub_1(5);
    subspace sub_2(5);
    std::vector<size_t> split_points_1;
    split_points_1.push_back(3);
    split_points_1.push_back(4);
    std::vector<size_t> split_points_2;
    split_points_2.push_back(2);
    split_points_2.push_back(4);
    sub_1.split(split_points_1); 
    sub_2.split(split_points_2); 

    if(sub_1 == sub_2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace::operator==(...) returned true");
    }
}

/* Should return zero
 */
void subspace_test::test_get_block_abs_index_one_block() throw(libtest::test_exception) {
    static const char *test_name = "subspace_test::test_get_block_abs_index_one_block()";
    subspace sub(5);
    if(! (sub.get_block_abs_index(0) == 0))
    {
        fail_test(test_name,__FILE__,__LINE__,
                "subspace::get_block_abs_index(0) did not return zero");
    }
}

/* Should return 4
 */
void subspace_test::test_get_block_abs_index_two_block() throw(libtest::test_exception) {
    static const char *test_name = "subspace_test::test_get_block_abs_index_one_block()";
    subspace sub(5);
    std::vector<size_t> split_points;
    split_points.push_back(4);
    sub.split(split_points);
    if(! (sub.get_block_abs_index(1) == 4))
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
    subspace sub(5);
    std::vector<size_t> split_points;
    split_points.push_back(4);
    sub.split(split_points);
    try
    {
        sub.get_block_abs_index(2);
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
#endif

} // namespace libtensor
