#ifndef LIBTENSOR_SO_SYMMETRIZE_TEST_H
#define LIBTENSOR_SO_SYMMETRIZE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::so_symmetrize class

    \ingroup libtensor_tests_sym
**/
class so_symmetrize_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();

};

} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_TEST_H
