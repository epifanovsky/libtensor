#ifndef LIBTENSOR_EVAL_SEQUENCE_LIST_TEST_H
#define LIBTENSOR_EVAL_SEQUENCE_LIST_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::evaluation_rule class

    \ingroup libtensor_tests_sym
 **/
class eval_sequence_list_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_EVAL_SEQUENCE_LIST_TEST_H

