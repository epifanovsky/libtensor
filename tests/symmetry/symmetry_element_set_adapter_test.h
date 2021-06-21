#ifndef LIBTENSOR_SYMMETRY_ELEMENT_SET_ADAPTER_TEST_H
#define LIBTENSOR_SYMMETRY_ELEMENT_SET_ADAPTER_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::symmetry_element_set_adapter class

    \ingroup libtensor_tests_sym
 **/
class symmetry_element_set_adapter_test : public libtest::unit_test {
public:
    virtual void perform();

private:
    void test_1();
    void test_2();

};

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_ELEMENT_SET_ADAPTER_TEST_H
