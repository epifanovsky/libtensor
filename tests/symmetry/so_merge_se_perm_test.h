#ifndef LIBTENSOR_SO_MERGE_SE_PERM_TEST_H
#define LIBTENSOR_SO_MERGE_SE_PERM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::so_merge_se_perm class

    \ingroup libtensor_tests_sym
 **/
class so_merge_se_perm_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_empty_1();
    void test_empty_2();
    void test_empty_3();
    void test_nn1(bool symm);
    void test_nm1_1(bool symm);
    void test_nm1_2(bool symm);
    void test_nm1_3(bool symm);
    void test_2n2nn_1(bool symm1, bool symm2);
    void test_2n2nn_2(bool symm);
    void test_nmk_1(bool symm);
    void test_nmk_2(bool symm);
};


} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_SE_PERM_TEST_H

