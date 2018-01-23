#ifndef LIBTENSOR_TO_COPY_WND_TEST_H
#define LIBTENSOR_TO_COPY_WND_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::to_copy_wnd class

    \ingroup libtensor_tests_tod
 **/
template<typename T>
class to_copy_wnd_test_x: public libtest::unit_test {
public:
    virtual void perform() throw (libtest::test_exception);

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();

};

class to_copy_wnd_test: public libtest::unit_test {
public:
    virtual void perform() throw (libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TO_COPY_WND_TEST_H

