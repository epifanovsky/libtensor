#ifndef LIBTENSOR_BLOCK_LABELING_TEST_H
#define LIBTENSOR_BLOCK_LABELING_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::block_labeling<N> class

    \ingroup libtensor_tests_sym
 **/
class block_labeling_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_basic_1();
    void test_basic_2();
    void test_copy_1();
    void test_permute_1();
    void test_transfer_1();
    void test_transfer_2();
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_LABELING_TEST_H

