#ifndef LIBTENSOR_SEQUENCE_GENERATOR_TEST_H
#define LIBTENSOR_SEQUENCE_GENERATOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::sequence_generator class

    \ingroup libtensor_tests_core
 **/
class sequence_generator_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_SEQUENCE_GENERATOR_TEST_H

