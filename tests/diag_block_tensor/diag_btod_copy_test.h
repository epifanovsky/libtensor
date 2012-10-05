#ifndef LIBTENSOR_DIAG_BTOD_COPY_TEST_H
#define LIBTENSOR_DIAG_BTOD_COPY_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::diag_btod_copy class

    \ingroup libtensor_diag_block_tensor_tests
 **/
class diag_btod_copy_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_copy_nosym_1();
    void test_copy_nosym_2();
    void test_copy_nosym_3();

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BTOD_COPY_TEST_H
