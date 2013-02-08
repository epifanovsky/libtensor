#ifndef LIBTENSOR_BTOD_SYMMETRIZE2_TEST_H
#define LIBTENSOR_BTOD_SYMMETRIZE2_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::btod_symmetrize2 class

    \ingroup libtensor_tests_btod
 **/
class btod_symmetrize2_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1() throw(libtest::test_exception);
    void test_2() throw(libtest::test_exception);
    void test_3() throw(libtest::test_exception);
    void test_4() throw(libtest::test_exception);
    void test_5(bool symm) throw(libtest::test_exception);
    void test_6a(bool symm, bool label,
            bool part, bool doadd) throw(libtest::test_exception);
    void test_6b(bool symm, bool label,
            bool part) throw(libtest::test_exception);
    void test_7();

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SYMMETRIZE2_TEST_H
