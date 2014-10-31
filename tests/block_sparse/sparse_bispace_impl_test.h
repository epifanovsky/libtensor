#ifndef SPARSE_BISPACE_IMPL_TEST_H
#define SPARSE_BISPACE_IMPL_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

class sparse_bispace_impl_test : public libtest::unit_test 
{
public:
    virtual void perform() throw(libtest::test_exception);
private:

    /*
     * operator==() tests
     */
    void test_equality_2d() throw(libtest::test_exception);
    void test_equality_2d_sparse() throw(libtest::test_exception);

    /*
     * sparse_bispace<N>::operator[](...) tests
     */
    //Negative tests (throw correct exception)
    void test_nd_subscript_operator_gt_upper_bound() throw(libtest::test_exception);

    /* 
     * sparse_bispace<N>::get_nnz(...) tests
     */
    //Positive tests (produce correct result)
    void test_get_nnz_dense() throw(libtest::test_exception);
    void test_get_nnz_2d_sparsity() throw(libtest::test_exception);
    void test_get_nnz_3d_dense_sparse() throw(libtest::test_exception);
    void test_get_nnz_3d_sparse_dense() throw(libtest::test_exception);
    void test_get_nnz_3d_fully_sparse() throw(libtest::test_exception);

    /*
     * sparse_bispace<N>::permute(...) tests
     */
    void test_permute_2d_10() throw(libtest::test_exception);
    void test_permute_3d_fully_sparse_210() throw(libtest::test_exception);
    void test_permute_3d_non_contiguous_sparsity() throw(libtest::test_exception);
    void test_permute_5d_sd_swap() throw(libtest::test_exception);
    void test_permute_5d_sd_interleave() throw(libtest::test_exception);

    /*
     * sparse_bispace<N>::contract(...) tests
     */
    void test_contract_3d_dense() throw(libtest::test_exception);
    void test_contract_3d_sparse_2() throw(libtest::test_exception);
    void test_contract_3d_sparse_2_nnz() throw(libtest::test_exception);
    void test_contract_3d_sparse_destroy_all_sparsity() throw(libtest::test_exception);

    void test_truncate_subspace() throw(libtest::test_exception);

    /*
     * sparse_bispace<N>::fuse(...) tests
     */
    void test_fuse_2d_2d() throw(libtest::test_exception);
    void test_fuse_3d_3d_no_overlap() throw(libtest::test_exception);
    void test_fuse_3d_3d_invalid_no_match() throw(libtest::test_exception);

    /*
     * index_group tests
     */
    void test_get_n_index_groups() throw(libtest::test_exception);
    void test_get_index_group_offset() throw(libtest::test_exception);
    void test_get_index_group_order() throw(libtest::test_exception);
    void test_get_index_group_dim() throw(libtest::test_exception);
    void test_get_index_group_containing_subspace() throw(libtest::test_exception);

    void test_get_batch_size() throw(libtest::test_exception);

};

} // namespace libtensor

#endif /* SPARSE_BISPACE_IMPL_TEST_H */
