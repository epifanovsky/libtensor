#ifndef SPARSE_BTENSOR_TEST_H
#define SPARSE_BTENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::sparse_btensor class

    \ingroup libtensor_tests_sparse
**/
class sparse_btensor_test : public libtest::unit_test { 
public:
    virtual void perform() throw(libtest::test_exception);
private:

    /** \brief Returns the correct bispace object that was used to construct it?
    **/
    void test_get_bispace() throw(libtest::test_exception);

    /** \brief Correct string representation of the tensor
     *         Loaded from block major memory
    **/
    void test_str_2d_block_major() throw(libtest::test_exception);

    /** \brief Correct string representation of the tensor
     *         Loaded from row major memory
    **/
    void test_str_2d_row_major() throw(libtest::test_exception);

    void test_str_3d_row_major() throw(libtest::test_exception);

    /** \brief Test that equality returns true when appropriate
    **/
    void test_equality_different_nnz() throw(libtest::test_exception);
    void test_equality_true() throw(libtest::test_exception);
    void test_equality_false() throw(libtest::test_exception);

    /** \brief Tensor is permuted correctly
    **/
    void test_permute_2d_row_major() throw(libtest::test_exception);
    void test_permute_3d_row_major_210() throw(libtest::test_exception);
    void test_permute_3d_block_major_120_sparse() throw(libtest::test_exception);

    /** \brief Sparse and dense matrix multiplies 
     **/
    void test_contract2_2d_2d() throw(libtest::test_exception);
    void test_contract2_2d_2d_sparse_dense() throw(libtest::test_exception);
    void test_contract2_3d_2d() throw(libtest::test_exception);
    void test_contract2_3d_2d_sparse_dense() throw(libtest::test_exception);
    void test_contract2_3d_2d_sparse_sparse() throw(libtest::test_exception);
    void test_contract2_two_indices_3d_3d_dense_dense() throw(libtest::test_exception);
    void test_contract2_two_indices_3d_3d_sparse_sparse() throw(libtest::test_exception);

    /** \brief Subtraction
     **/
    void test_subtraction_2d_2d() throw(libtest::test_exception);

    //TODO: REMOVE
    void test_performance();
};

} // namespace libtensor

#endif /* SPARSE_BTENSOR_TEST_H */
