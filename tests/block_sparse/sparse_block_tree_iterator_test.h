#ifndef SPARSE_BLOCK_TREE_ITERATOR_TEST_H
#define SPARSE_BLOCK_TREE_ITERATOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {
    
/** \brief Tests the libtensor::sparse_block_tree
 
 \ingroup libtensor_tests_sparse
 **/
class sparse_block_tree_iterator_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
private:
    /** \brief Should give correct key and value for initial node
     **/
    void test_begin_3d() throw(libtest::test_exception);
    
    /** \brief For an empty tree, end() should '==' begin() 
     **/
    void test_end_3d() throw(libtest::test_exception);

    /** \brief Test that increment can produce correct iterator
     **/
    void test_incr_3d() throw(libtest::test_exception);
};
    
} // namespace libtensor

#endif /* SPARSE_BLOCK_TREE_TEST_H */
