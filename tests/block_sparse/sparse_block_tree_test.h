#ifndef SPARSE_BLOCK_TREE_TEST_H
#define SPARSE_BLOCK_TREE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor { 

/** \brief Tests the libtensor::sparse_block_tree

    \ingroup libtensor_tests_sparse
**/
class sparse_block_tree_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
private:

    /** \brief Should throw an exception when a zero-order tree requested
     **/
    void test_zero_order() throw(libtest::test_exception);
    /** \brief Should throw an exception if sig block lists unsorted
    **/
    void test_unsorted_input() throw(libtest::test_exception);

    /** \brief Test that two things that aren't equal are shown as such 
    **/
    void test_equality_false_2d() throw(libtest::test_exception);
    /** \brief Test that two things that are equal are shown as such 
    **/
    void test_equality_true_2d() throw(libtest::test_exception);



    /** \brief Test that search returns end() when an invalid key is specified 
    **/
    void test_search_2d_invalid_key() throw(libtest::test_exception);

    /** \brief Test that search returns an iterator to the correct value when searching a 3d tree
    **/
    void test_search_3d() throw(libtest::test_exception);


    /** \brief Ensures that exception thrown when key is too big or empty 
    **/
    void test_get_sub_tree_invalid_key_size() throw(libtest::test_exception);

    /** \brief Ensures that method returns an empty tree when key does not exist 
    **/
    void test_get_sub_tree_nonexistent_key() throw(libtest::test_exception);
    
    /** \brief Ensures that return value contains all correct keys for 2d
    **/
    void test_get_sub_tree_2d() throw(libtest::test_exception);
 
    /** \brief Ensures that return value contains all correct keys for 3d
    **/
    void test_get_sub_tree_3d() throw(libtest::test_exception);

    /** \brief Test that permutation of tree produces correct result 
    **/
    void test_permute_3d() throw(libtest::test_exception);



    /** \brief Ensure that contracting a 3d tree gives the correct result for all dimensions
     **/
    void test_contract_3d_0() throw(libtest::test_exception);
    void test_contract_3d_1() throw(libtest::test_exception);
    void test_contract_3d_2() throw(libtest::test_exception);

    /** \brief Fuses multiple trees into one
     **/
    void test_fuse_2d_2d() throw(libtest::test_exception);
    void test_fuse_3d_2d() throw(libtest::test_exception);
    void test_fuse_3d_3d_non_contiguous() throw(libtest::test_exception);
    void test_fuse_3d_3d_multi_index() throw(libtest::test_exception);
};

} // namespace libtensor

#endif /* SPARSE_BLOCK_TREE_TEST_H */
