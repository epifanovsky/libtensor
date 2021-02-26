#ifndef LIBTENSOR_TO_SET_TEST_H
#define LIBTENSOR_TO_SET_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::to_set class

    \ingroup libtensor_tests_tod
**/
class to_set_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    template<typename T>
    void test_1(T) throw(libtest::test_exception);

    template<typename T>
    void test_2(T) throw(libtest::test_exception);

    template<typename T>
    void test_3(T) throw(libtest::test_exception);

    template<typename T>
    void test_4(T) throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_TO_SET_TEST_H

