#ifndef LIBTENSOR_TOD_CONV_DIAG_TENSOR_TEST_H
#define LIBTENSOR_TOD_CONV_DIAG_TENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::tod_conv_diag_tensor class

    \ingroup libtensor_diag_tensor_tests
 **/
class tod_conv_diag_tensor_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1_i(size_t ni) throw(libtest::test_exception);
    void test_2_i(size_t ni) throw(libtest::test_exception);
    void test_3_i(size_t ni) throw(libtest::test_exception);
    void test_1_ij(size_t ni, size_t nj) throw(libtest::test_exception);
    void test_2_ij(size_t ni, size_t nj) throw(libtest::test_exception);
    void test_1_ii(size_t ni) throw(libtest::test_exception);
    void test_1_ii_ij(size_t ni) throw(libtest::test_exception);
    void test_1_iik_iji(size_t ni) throw(libtest::test_exception);
    void test_1_iijj_ijij_ijjk(size_t ni) throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_CONV_DIAG_TENSOR_TEST_H

