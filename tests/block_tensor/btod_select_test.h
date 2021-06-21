#ifndef LIBTENSOR_BTOD_SELECT_TEST_H
#define LIBTENSOR_BTOD_SELECT_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::btod_select class

    \ingroup libtensor_tests_btod
**/
class btod_select_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    template<typename ComparePolicy>
    void test_1(size_t n);
    template<typename ComparePolicy>
    void test_2(size_t n);
    template<typename ComparePolicy>
    void test_3a(size_t n, bool symm);
    template<typename ComparePolicy>
    void test_3b(size_t n);
    template<typename ComparePolicy>
    void test_3c(size_t n, bool symm);
    template<typename ComparePolicy>
    void test_4a(size_t n, bool symm);
    template<typename ComparePolicy>
    void test_4b(size_t n);
    template<typename ComparePolicy>
    void test_4c(size_t n, bool symm);
    template<typename ComparePolicy>
    void test_5(size_t n);
};

} // namespace libtensor

#endif // LIBTENSOR_BTOD_SELECT_TEST_H
