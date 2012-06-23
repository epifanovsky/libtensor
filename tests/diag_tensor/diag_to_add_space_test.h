#ifndef LIBTENSOR_DIAG_TO_ADD_SPACE_TEST_H
#define LIBTENSOR_DIAG_TO_ADD_SPACE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::diag_to_add_space class

    \ingroup libtensor_diag_tensor_tests
 **/
class diag_to_add_space_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TO_ADD_SPACE_TEST_H

