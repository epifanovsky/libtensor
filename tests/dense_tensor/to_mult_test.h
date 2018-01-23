#ifndef LIBTENSOR_TO_MULT_TEST_H
#define LIBTENSOR_TO_MULT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::to_mult class

    \ingroup libtensor_tests_tod
**/
template<typename T>
class to_mult_test_x : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
    const static T k_thresh;
private:
    void test_pq_pq_1(size_t ni, size_t nj,
            bool recip) throw(libtest::test_exception);
    void test_pq_pq_2(size_t ni, size_t nj,
            bool recip, T coeff) throw(libtest::test_exception);
    void test_pq_qp(bool recip, T coeff) throw(libtest::test_exception);
    void test_qp_pq(bool recip, T coeff) throw(libtest::test_exception);
    void test_qp_qp(bool recip, T coeff) throw(libtest::test_exception);
    void test_pqrs_qprs(size_t ni, size_t nj, size_t nk, size_t nl,
            bool recip, T coeff) throw(libtest::test_exception);
    void test_pqrs_qrps(size_t ni, size_t nj, size_t nk, size_t nl,
            bool recip, T coeff) throw(libtest::test_exception);
    void test_pqsr_pqrs(size_t ni, size_t nj, size_t nk, size_t nl,
            bool recip, T coeff) throw(libtest::test_exception);
    void test_prsq_qrps(size_t ni, size_t nj, size_t nk, size_t nl,
            bool recip, T coeff) throw(libtest::test_exception);
};

class to_mult_test : public libtest::unit_test  {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TO_MULT_TEST_H
