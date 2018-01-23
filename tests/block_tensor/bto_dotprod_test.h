#ifndef LIBTENSOR_BTO_DOTPROD_TEST_H
#define LIBTENSOR_BTO_DOTPROD_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests the libtensor::bto_dotprod class

    \ingroup libtensor_tests_btod
**/
template<typename T>
class bto_dotprod_test_x : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
    static const T k_thresh; //!< Threshold multiplier

private:
    void test_1() throw(libtest::test_exception);
    void test_2() throw(libtest::test_exception);
    void test_3() throw(libtest::test_exception);
    void test_4() throw(libtest::test_exception);
    void test_5() throw(libtest::test_exception);
    void test_6() throw(libtest::test_exception);
    void test_7() throw(libtest::test_exception);
    void test_8() throw(libtest::test_exception);
    void test_9() throw(libtest::test_exception);
    void test_10a() throw(libtest::test_exception);
    void test_10b() throw(libtest::test_exception);
    void test_10c(bool both) throw(libtest::test_exception);
    void test_11() throw(libtest::test_exception);
    void test_12() throw(libtest::test_exception);
    void test_13a() throw(libtest::test_exception);
    void test_13b() throw(libtest::test_exception);

};

class bto_dotprod_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTO_DOTPROD_TEST_H
