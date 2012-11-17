#ifndef LIBTENSOR_DIAG_TOD_CONTRACT2_TEST_H
#define LIBTENSOR_DIAG_TOD_CONTRACT2_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::diag_tod_contract2 class

    \ingroup libtensor_diag_tensor_tests
 **/
class diag_tod_contract2_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1_1_1_01(size_t ni, size_t nj, size_t nk);
    void test_1_1_1_02(size_t ni, size_t nj, size_t nk);
    void test_1_1_1_03(size_t ni, size_t nj);
    void test_1_1_1_04(size_t ni);
    void test_1_1_1_05(size_t ni);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_CONTRACT2_TEST_H

