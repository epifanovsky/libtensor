#ifndef BATCH_KERNELS_TEST_H
#define BATCH_KERNELS_TEST_H

#include <libtest/unit_test.h>

namespace libtensor
{

class batch_kernels_test : public libtest::unit_test
{
public:
    virtual void perform() throw(libtest::test_exception);
private:
    void test_batch_kernel_permute_A_direct() throw(libtest::test_exception);
    void test_batch_kernel_contract2() throw(libtest::test_exception);
    void test_batch_kernel_add2() throw(libtest::test_exception);
};

} /* namespace libtensor */

#endif /* BATCH_KERNELS_TEST_H */
