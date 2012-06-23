#ifndef LIBTENSOR_LETTER_TEST_H
#define LIBTENSOR_LETTER_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::letter class

    \ingroup libtensor_tests_iface
**/
class letter_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

};

} // namespace libtensor

#endif // LIBTENSOR_LETTER_TEST_H

