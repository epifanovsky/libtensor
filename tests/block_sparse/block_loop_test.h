#ifndef BLOCK_LOOP_TEST_H
#define BLOCK_LOOP_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

class block_loop_test : public libtest::unit_test { 
public:
    virtual void perform() throw(libtest::test_exception);
private: 

    void test_range() throw(libtest::test_exception);


	void test_run_invalid_bispaces() throw(libtest::test_exception);
	/*
	 * Copying
	 */
    void test_run_block_copy_kernel_1d() throw(libtest::test_exception);
    void test_run_block_copy_kernel_2d() throw(libtest::test_exception);

	/*
	 * Permutation
	 */
    void test_run_block_permute_kernel_2d() throw(libtest::test_exception);
    void test_run_block_permute_kernel_2d_sparse() throw(libtest::test_exception);
    void test_run_block_permute_kernel_3d_201() throw(libtest::test_exception);
    void test_run_block_permute_kernel_3d_201_sparse() throw(libtest::test_exception);

    /*
     * Contraction
     */
    void test_run_block_contract2_kernel_2d_2d() throw(libtest::test_exception);
    void test_run_block_contract2_kernel_3d_2d() throw(libtest::test_exception);
};

} // namespace libtensor

#endif /* BLOCK_LOOP_TEST_H */
