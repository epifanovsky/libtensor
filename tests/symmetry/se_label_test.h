#ifndef LIBTENSOR_SE_LABEL_TEST_H
#define LIBTENSOR_SE_LABEL_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/**	\brief Tests the libtensor::se_label class

	\ingroup libtensor_tests_sym
 **/
class se_label_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    static const char *table_id;

    // Empty
    void test_empty() throw(libtest::test_exception);
    // One subset, full mask
    void test_set_1() throw(libtest::test_exception);
    // One subset, partial mask
    void test_set_2() throw(libtest::test_exception);
    // Two subsets, full mask
    void test_set_3() throw(libtest::test_exception);
    // Two subsets, partial mask
    void test_set_4() throw(libtest::test_exception);
    void test_permute() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_SE_LABEL_TEST_H

