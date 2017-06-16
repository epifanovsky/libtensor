#ifndef LIBTENSOR_BTO_SET_TEST_H
#define LIBTENSOR_BTO_SET_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_set class

    \ingroup libtensor_tests_btod
**/
template<typename T>
class bto_set_test_x : public libtest::unit_test {
public:
    static const T k_thresh;
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);
    void test_2() throw(libtest::test_exception);
};

class bto_set_test : public libtest::unit_test  {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTO_SET_TEST_H
