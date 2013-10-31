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
    /** \brief Should throw an exception if sig block lists unsorted
    **/
    void test_unsorted_input() throw(libtest::test_exception);



    /** \brief Ensures that get_sub_key_iterator throws exception when key is too big or empty 
    **/
    void test_get_sub_key_iterator_invalid_key_size() throw(libtest::test_exception);

    /** \brief Ensures that get_sub_key_iterator throws exception when key does not exist 
    **/
    void test_get_sub_key_iterator_nonexistent_key() throw(libtest::test_exception);
    
    /** \brief Ensures that get_sub_key_iterator can be incremented to produce all correct values
    **/
    void test_get_sub_key_iterator_2d() throw(libtest::test_exception);


    /** \brief Test that iterator key method returns the correct value for the first method 
    **/
    void test_iterator_2d_key() throw(libtest::test_exception);

    /** \brief Ensures that iterating over the container produces the correct keys and values
    **/
    void test_iterator_2d_incr() throw(libtest::test_exception);

    /** \brief Ensures that all of the container values can be set appropriately 
    **/
    void test_iterator_2d_set() throw(libtest::test_exception);

    /** \brief Ensures that iterating over the container produces the correct keys and values for a 3d case
    **/
    void test_iterator_3d_incr() throw(libtest::test_exception);

    /** \brief Test that search throws an exception when an invalid key is specified 
    **/
    void test_search_2d_invalid_key() throw(libtest::test_exception);

    /** \brief Test that search returns an iterator to the correct value when searching a 2d tree
    **/
    void test_search_2d() throw(libtest::test_exception);

    /** \brief Test that search returns an iterator to the correct value when searching a 3d tree
    **/
    void test_search_3d() throw(libtest::test_exception);


    /** \brief Test that permutation of tree produces correct result 
    **/
    void test_permute_2d() throw(libtest::test_exception);

    /** \brief Test that permutation of tree produces correct result 
    **/
    void test_permute_3d() throw(libtest::test_exception);

    /** \brief Test that two things that aren't equal are shown as such 
    **/
    void test_equality_false_2d() throw(libtest::test_exception);

    /** \brief Test that two things that are equal are shown as such 
    **/
    void test_equality_true_2d() throw(libtest::test_exception);
};

} // namespace libtensor

#endif /* SPARSE_BLOCK_TREE_TEST_H */
