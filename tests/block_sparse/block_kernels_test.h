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
    void test_block_printer_2d() throw(libtest::test_exception);
    void test_block_printer_3d() throw(libtest::test_exception);

    /*
     * block_copy_kernel tests
     */
    void test_block_copy_kernel_2d() throw(libtest::test_exception);
    void test_block_copy_kernel_3d() throw(libtest::test_exception);

    void test_block_equality_kernel_2d_true() throw(libtest::test_exception);
    void test_block_equality_kernel_2d_false() throw(libtest::test_exception);
    void test_block_equality_kernel_not_run_exception() throw(libtest::test_exception);
};

} // namespace libtensor

#endif /* BLOCK_KERNELS_TEST_H */
