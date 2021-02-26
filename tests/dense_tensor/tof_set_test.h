#ifndef LIBTENSOR_TOF_SET_TEST_H
#define LIBTENSOR_TOF_SET_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::tof_set class

    \ingroup libtensor_tests_tod
**/
class tof_set_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1(float d) throw(libtest::test_exception);
    void test_2(float d) throw(libtest::test_exception);
    void test_3(float d) throw(libtest::test_exception);
    void test_4(float d) throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_TOF_SET_TEST_H

