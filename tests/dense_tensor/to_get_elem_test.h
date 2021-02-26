#ifndef LIBTENSOR_TO_GET_ELEM_TEST_H
#define LIBTENSOR_TO_GET_ELEM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::to_get_elem class

    \ingroup libtensor_tests_tod
**/
template<typename T>
class to_get_elem_test_x : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);
};

class to_get_elem_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TO_GET_ELEM_TEST_H
