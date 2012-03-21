#ifndef LIBTENSOR_EVALUATION_RULE_TEST_H
#define LIBTENSOR_EVALUATION_RULE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::evaluation_rule class

	\ingroup libtensor_tests_sym
 **/
class evaluation_rule_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);
    void test_2() throw(libtest::test_exception);
    void test_opt_1() throw(libtest::test_exception);
    void test_opt_2() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_EVALUATION_RULE_TEST_H

