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

    /** \brief Ensures that iterating over the container produces the correct key/value pairs 
    **/
    void test_iterator_2d() throw(libtest::test_exception);
    //TODO: Need a 3d!!!
};

} // namespace libtensor

#endif /* SPARSE_BLOCK_TREE_TEST_H */
