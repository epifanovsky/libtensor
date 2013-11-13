#ifndef BLOCK_KERNELS_TEST_H
#define BLOCK_KERNELS_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

class block_kernels_test : public libtest::unit_test { 
public:
    virtual void perform() throw(libtest::test_exception);
private: 

    /*
     * block_printer tests
     */
    void test_block_print_kernel_2d() throw(libtest::test_exception);
    void test_block_print_kernel_3d() throw(libtest::test_exception);

    /*
     * block_permute_kernel tests
     */
    void test_block_permute_kernel_2d() throw(libtest::test_exception);
    void test_block_permute_kernel_3d_120() throw(libtest::test_exception);
    void test_block_permute_kernel_3d_021() throw(libtest::test_exception);

    /*
     * block_contract_kernel tests
     */
    void test_block_contract2_kernel_2d_not_enough_indices() throw(libtest::test_exception);
    void test_block_contract2_kernel_2d_strided_output() throw(libtest::test_exception);
    void test_block_contract2_kernel_2d_oob_indices() throw(libtest::test_exception);
    void test_block_contract2_kernel_2d_not_matching_indices() throw(libtest::test_exception);
    void test_block_contract2_kernel_2d_wrong_dim_order() throw(libtest::test_exception);

    void test_block_contract2_kernel_2d_ip_pj() throw(libtest::test_exception);
    void test_block_contract2_kernel_2d_ip_jp() throw(libtest::test_exception);
    void test_block_contract2_kernel_2d_pi_pj() throw(libtest::test_exception);
    void test_block_contract2_kernel_2d_pi_jp() throw(libtest::test_exception);

    void test_block_contract2_kernel_3d_2d() throw(libtest::test_exception);
    void test_block_contract2_kernel_3d_3d_multi_index() throw(libtest::test_exception);

    /*
     * block_subtract_kernel tests
     */
    void test_block_subtract_kernel_2d_2d() throw(libtest::test_exception);

    /*
     * direct_kernel tests
     */
    void test_direct_block_subtract_kernel_2d_2d() throw(libtest::test_exception);
};


} // namespace libtensor

#endif /* BLOCK_KERNELS_TEST_H */
