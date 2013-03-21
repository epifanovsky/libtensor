#ifndef LIBTENSOR_DIAG_TOD_DOTPROD_TEST_H
#define LIBTENSOR_DIAG_TOD_DOTPROD_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::diag_tod_dotprod class

    \ingroup libtensor_diag_tensor_tests
 **/
class diag_tod_dotprod_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1(size_t ni, size_t nj);
    void test_2(size_t ni, size_t nj);
    void test_3(size_t ni);
    void test_4(size_t ni, size_t nj);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_DOTPROD_TEST_H

