#ifndef LIBTENSOR_SO_PERMUTE_SE_PART_TEST_H
#define LIBTENSOR_SO_PERMUTE_SE_PART_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::so_permute_se_part class

    \ingroup libtensor_tests_sym
 **/
class so_permute_se_part_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2a();
    void test_2b();
    void test_3();

};


} // namespace libtensor

#endif // LIBTENSOR_SO_PERMUTE_SE_PART_TEST_H
