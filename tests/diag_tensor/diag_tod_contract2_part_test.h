#ifndef LIBTENSOR_DIAG_TOD_CONTRACT2_PART_TEST_H
#define LIBTENSOR_DIAG_TOD_CONTRACT2_PART_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::diag_tod_contract2_part class

    \ingroup libtensor_diag_tensor_tests
 **/
class diag_tod_contract2_part_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_ij_ik_kj(size_t ni, size_t nj, size_t nk)
        throw(libtest::test_exception);
    void test_ij_ii_ij(size_t ni, size_t nj) throw(libtest::test_exception);
    void test_ii_ii_ij(size_t ni) throw(libtest::test_exception);
    void test_ii_ii_ii(size_t ni) throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_CONTRACT2_PART_TEST_H

