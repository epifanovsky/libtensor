#ifndef LIBTENSOR_ABS_INDEX_TEST_H
#define LIBTENSOR_ABS_INDEX_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the libtensor::abs_index class

    \ingroup libtensor_tests_core
 **/
class abs_index_test: public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_ctor_1() throw(libtest::test_exception);
    void test_ctor_2() throw(libtest::test_exception);
    void test_ctor_3() throw(libtest::test_exception);
    void test_ctor_4() throw(libtest::test_exception);
    void test_ctor_5() throw(libtest::test_exception);
    void test_inc_1() throw(libtest::test_exception);
    void test_inc_2() throw(libtest::test_exception);
    void test_inc_3() throw(libtest::test_exception);
    void test_inc_4() throw(libtest::test_exception);
    void test_inc_5() throw(libtest::test_exception);
    void test_last_1() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_ABS_INDEX_TEST_H
