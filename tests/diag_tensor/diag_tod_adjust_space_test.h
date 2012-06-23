#ifndef LIBTENSOR_DIAG_TOD_ADJUST_SPACE_TEST_H
#define LIBTENSOR_DIAG_TOD_ADJUST_SPACE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::diag_tod_adjust_space class

    \ingroup libtensor_diag_tensor_tests
 **/
class diag_tod_adjust_space_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_ai_bi(size_t ni) throw(libtest::test_exception);
    void test_ai_ai_bi(size_t ni) throw(libtest::test_exception);
    void test_aijk_biij_biii(size_t ni) throw(libtest::test_exception);
    void test_aiii_aijk_biii_biij(size_t ni) throw(libtest::test_exception);
    void test_aijkl_biiii(size_t ni) throw(libtest::test_exception);
    void test_aijkl_biiii_bijkl(size_t ni) throw(libtest::test_exception);
    void test_aiiii_biijj_bijij(size_t ni) throw(libtest::test_exception);
    void test_aiiii_aijij_biijj_bijij(size_t ni) throw(libtest::test_exception);
    void test_aiijj_biiij(size_t ni) throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_ADJUST_SPACE_TEST_H

