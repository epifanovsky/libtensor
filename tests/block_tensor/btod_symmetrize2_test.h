#ifndef LIBTENSOR_BTOD_SYMMETRIZE2_TEST_H
#define LIBTENSOR_BTOD_SYMMETRIZE2_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::btod_symmetrize2 class

    \ingroup libtensor_tests_btod
 **/
class btod_symmetrize2_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5(bool symm);
    void test_6a(bool symm, bool label,
            bool part, bool doadd);
    void test_6b(bool symm, bool label,
            bool part);
    void test_7();

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SYMMETRIZE2_TEST_H
