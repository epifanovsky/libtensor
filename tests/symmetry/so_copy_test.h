#ifndef LIBTENSOR_SO_COPY_TEST_H
#define LIBTENSOR_SO_COPY_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::so_copy class

    \ingroup libtensor_tests_sym
**/
class so_copy_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();
    void test_3();

};

} // namespace libtensor

#endif // LIBTENSOR_SO_COPY_TEST_H
