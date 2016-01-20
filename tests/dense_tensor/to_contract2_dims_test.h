#ifndef LIBTENSOR_TO_CONTRACT2_DIMS_TEST_H
#define LIBTENSOR_TO_CONTRACT2_DIMS_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::to_contract2_dims class

    \ingroup libtensor_tests_tod
**/
class to_contract2_dims_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_ij_i_j(size_t ni, size_t nj) throw(libtest::test_exception);
    void test_ij_j_i(size_t ni, size_t nj) throw(libtest::test_exception);
    void test_ij_ik_jk(size_t ni, size_t nj, size_t nk)
        throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_TO_CONTRACT2_DIMS_TEST_H

