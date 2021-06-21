#ifndef LIBTENSOR_ADDITION_SCHEDULE_TEST_H
#define LIBTENSOR_ADDITION_SCHEDULE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::addition_schedule class

    \ingroup libtensor_tests_btod
 **/
class addition_schedule_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();
    void test_6();
    void test_7();
    void test_8();

};

} // namespace libtensor

#endif // LIBTENSOR_ADDITION_SCHEDULE_TEST_H
