#ifndef LIBTENSOR_SPARSE_BISPACE_TEST_H 
#define LIBTENSOR_SPARSE_BISPACE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::sparse_bispace class

    \ingroup libtensor_tests_sparse
 **/
class sparse_bispace_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:

    /* 
     * sparse_bispace<1>::get_dim(...) tests
     */
    //Positive tests (produce correct result)
    void test_get_dim() throw(libtest::test_exception);

    /* 
     * sparse_bispace<1>::get_n_blocks(...) tests
     */
    //Positive tests (produce correct result)
    void test_get_n_blocks() throw(libtest::test_exception);


    /* 
     * sparse_bispace<1>::split(...) tests
     */

    //Negative tests (throw correct exception) 
    void test_split_not_strictly_increasing() throw(libtest::test_exception);
    void test_split_not_strictly_increasing_two_calls() throw(libtest::test_exception);
    void test_split_gt_upper_bound() throw(libtest::test_exception);
    void test_split_zero_size() throw(libtest::test_exception);

    /* 
     * sparse_bispace<1>::operator==(...) tests
     */
    //Positive tests (produce correct result)
    void test_equality_true() throw(libtest::test_exception);
    void test_equality_false_diff_dims() throw(libtest::test_exception);
    void test_equality_false_diff_splits() throw(libtest::test_exception);
    void test_nd_equality_true() throw(libtest::test_exception);

    /* 
     * sparse_bispace<1>::get_block_abs_index(...) tests
     */

    //Positive tests (produce correct result)
    void test_get_block_abs_index_one_block() throw(libtest::test_exception);
    void test_get_block_abs_index_two_block() throw(libtest::test_exception);
    
    //Negative tests (throw correct exception)
    void test_get_block_abs_index_gt_upper_bound() throw(libtest::test_exception);


    /* 
     * sparse_bispace<1>::get_block_size(...) tests
     */

    //Positive tests (produce correct result)
    void test_get_block_size_one_block() throw(libtest::test_exception);
    void test_get_block_size_two_block() throw(libtest::test_exception);
    
    //Negative tests (throw correct exception)
    void test_get_block_size_gt_upper_bound() throw(libtest::test_exception);  

    /*
     * sparse_bispace<N>::operator[](...) tests
     */
    //Negative tests (throw correct exception)
    void test_nd_subscript_operator_gt_upper_bound() throw(libtest::test_exception);

    /*
     * sparse_bispace<N>::operator|(...) tests
     */
    //Positive tests (produce correct result)
    void test_nd_bar_operator_both_operands_1d() throw(libtest::test_exception);
    void test_nd_bar_operator_both_operands_2d() throw(libtest::test_exception);

    /* 
     * sparse_bispace<N>::get_nnz(...) tests
     */
    //Positive tests (produce correct result)
    void test_get_nnz_dense() throw(libtest::test_exception);

    /* 
     * sparse_bispace<N>::get_block_offset(...)  tests
     */
    void test_get_block_offset_1d() throw(libtest::test_exception);
    void test_get_block_offset_1d_empty_vec() throw(libtest::test_exception);
    void test_get_block_offset_1d_oob() throw(libtest::test_exception);
    void test_get_block_offset_2d() throw(libtest::test_exception);

    /* 
     * sparse_bispace<N>::get_block_offset_canonical(...)  tests
     */
    void test_get_block_offset_canonical_1d() throw(libtest::test_exception);
    void test_get_block_offset_canonical_1d_empty_vec() throw(libtest::test_exception);
    void test_get_block_offset_canonical_1d_oob() throw(libtest::test_exception);
    void test_get_block_offset_canonical_2d() throw(libtest::test_exception);

    /*
     * sparse_bispace<N>::permute(...) tests
     */
    void test_permute_2d_10() throw(libtest::test_exception);
    void test_permute_3d_dense_sparse_021() throw(libtest::test_exception);
    void test_permute_3d_fully_sparse_210() throw(libtest::test_exception);

    /*
     * sparse_bispace<N>::contract(...) tests
     */
    void test_contract_3d_dense() throw(libtest::test_exception);
    void test_contract_3d_sparse_2() throw(libtest::test_exception);
    void test_contract_3d_sparse_destroy_all_sparsity() throw(libtest::test_exception);


    /*  ALL TESTS INVOLVING SPARSITY!!
     *
     */
    void test_get_nnz_2d_sparsity() throw(libtest::test_exception);
    void test_get_nnz_3d_dense_sparse() throw(libtest::test_exception);
    void test_get_nnz_3d_sparse_dense() throw(libtest::test_exception);
    void test_get_nnz_3d_fully_sparse() throw(libtest::test_exception);
    void test_get_block_offset_2d_sparsity() throw(libtest::test_exception);
    void test_get_block_offset_3d_dense_sparse() throw(libtest::test_exception);
    void test_get_block_offset_3d_sparse_dense() throw(libtest::test_exception);
    void test_get_block_offset_3d_fully_sparse() throw(libtest::test_exception);
    void test_equality_false_sparsity_2d() throw(libtest::test_exception);
    void test_equality_true_sparsity_2d() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_SPARSE_BISPACE_TEST_H
