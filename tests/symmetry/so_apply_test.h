#ifndef LIBTENSOR_SO_APPLY_TEST_H
#define LIBTENSOR_SO_APPLY_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::so_apply class

    \ingroup libtensor_tests_sym
**/
class so_apply_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1(bool keep_zero,
            bool is_asym, bool sign);
    void test_2(bool keep_zero,
            bool is_asym, bool sign);
    void test_3(bool keep_zero,
            bool is_asym, bool sign);

};

} // namespace libtensor

#endif // LIBTENSOR_SO_APPLY_TEST_H
