#ifndef LIBTENSOR_DIAG_TOD_COPY_TEST_H
#define LIBTENSOR_DIAG_TOD_COPY_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::diag_tod_copy class

    \ingroup libtensor_diag_tensor_tests
 **/
class diag_tod_copy_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_ij_1(size_t ni, size_t nj, double d)
        throw(libtest::test_exception);
    void test_ij_2(size_t ni, size_t nj, double d)
        throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_COPY_TEST_H

