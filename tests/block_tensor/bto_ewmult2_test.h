#ifndef LIBTENSOR_BTO_EWMULT2_TEST_H
#define LIBTENSOR_BTO_EWMULT2_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::bto_ewmult2 class

    \ingroup libtensor_tests
 **/
template<typename T>
class bto_ewmult2_test_x : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
    static const T k_thresh; //!< Threshold multiplier

private:
    void test_1(bool doadd) throw(libtest::test_exception);
    void test_2(bool doadd) throw(libtest::test_exception);
    void test_3(bool doadd) throw(libtest::test_exception);
    void test_4(bool doadd) throw(libtest::test_exception);
    void test_5(bool doadd) throw(libtest::test_exception);
    void test_6(bool doadd) throw(libtest::test_exception);
    void test_7() throw(libtest::test_exception);

};

class bto_ewmult2_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTO_EWMULT2_TEST_H
