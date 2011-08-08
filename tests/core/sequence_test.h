#ifndef LIBTENSOR_SEQUENCE_TEST_H
#define LIBTENSOR_SEQUENCE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the libtensor::sequence class

    \sa sequence<N, T>

    \ingroup libtensor_tests_core
 **/
class sequence_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_ctor_1() throw(libtest::test_exception);
    void test_ctor_2() throw(libtest::test_exception);
    void test_ctor_3() throw(libtest::test_exception);
    void test_ctor_4() throw(libtest::test_exception);
    void test_ctor_5() throw(libtest::test_exception);
    void test_ctor_6() throw(libtest::test_exception);
    void test_ctor_7() throw(libtest::test_exception);
    void test_ctor_8() throw(libtest::test_exception);
    void test_ctor_9() throw(libtest::test_exception);

    void test_at_1() throw(libtest::test_exception);
    void test_at_2() throw(libtest::test_exception);

    void test_exc_1() throw(libtest::test_exception);
    void test_exc_2() throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_SEQUENCE_TEST_H
