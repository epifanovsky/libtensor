#ifndef LIBTENSOR_LABEL_SET_TEST_H
#define LIBTENSOR_LABEL_SET_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::label_set<N> class

	\ingroup libtensor_tests_sym
 **/
class label_set_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    static const char *table_id;

    void test_basic_1() throw(libtest::test_exception);
    void test_basic_2() throw(libtest::test_exception);
    void test_set_1() throw(libtest::test_exception);
    void test_set_2() throw(libtest::test_exception);
    void test_set_3() throw(libtest::test_exception);
    void test_set_4() throw(libtest::test_exception);
    void test_permute_1() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_LABEL_SET_TEST_H

