#ifndef LIBTENSOR_TOD_SCATTER_TEST_H
#define LIBTENSOR_TOD_SCATTER_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::tod_scatter class

    \ingroup libtensor_tests_tod
**/
class tod_scatter_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    // c_{ij} = a_j
    void test_ij_j(size_t ni, size_t nj, double d = 0.0)
        throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_TOD_SCATTER_TEST_H

