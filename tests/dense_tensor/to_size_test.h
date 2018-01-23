#ifndef LIBTENSOR_TO_SIZE_TEST_H
#define LIBTENSOR_TO_SIZE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::to_size class

    \ingroup libtensor_tests_tod
**/
template<typename T>
class to_size_test_x : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();

};

class to_size_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_TO_SIZE_TEST_H

