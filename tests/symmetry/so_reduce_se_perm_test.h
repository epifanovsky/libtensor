#ifndef LIBTENSOR_SO_REDUCE_SE_PERM_TEST_H
#define LIBTENSOR_SO_REDUCE_SE_PERM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the class
        libtensor::so_operation_impl<
            libtensor::so_reduce<N, M, K, T>, libtensor::se_perm<N, T> >

    \ingroup libtensor_tests_sym
 **/
class so_reduce_se_perm_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_empty_1();
    void test_empty_2();
    void test_nm1_1(bool symm);
    void test_nm1_2(bool symm);
    void test_nmk_1(bool symm);
    void test_nmk_2(bool symm);
    void test_nmk_3(bool symm1, bool symm2);

};


} // namespace libtensor

#endif // LIBTENSOR_SO_REDUCE_SE_PERM_TEST_H

