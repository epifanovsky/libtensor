#ifndef SPARSITY_DATA_TEST_H
#define SPARSITY_DATA_TEST_H

#include <libtest/unit_test.h>

namespace libtensor { 

/** \brief Tests the libtensor::sparsity_data class

    \ingroup libtensor_tests_sparse
**/
class sparsity_data_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
private:

    void test_zero_order() throw(libtest::test_exception);
    void test_invalid_key_length() throw(libtest::test_exception);
    void test_unsorted_keys() throw(libtest::test_exception);
    void test_duplicate_keys() throw(libtest::test_exception);

    void test_equality_2d() throw(libtest::test_exception);

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
    void test_fuse_3d_2d() throw(libtest::test_exception);
    void test_fuse_3d_3d_non_contiguous() throw(libtest::test_exception);
    void test_fuse_3d_3d_multi_index() throw(libtest::test_exception);

    void test_truncate_subspace_3d() throw(libtest::test_exception);

    void test_insert_subspace_3d() throw(libtest::test_exception);
};

} // namespace libtensor

#endif /* SPARSITY_DATA_TEST_H */
