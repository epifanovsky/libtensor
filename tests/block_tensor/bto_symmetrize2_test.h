#ifndef LIBTENSOR_BTO_SYMMETRIZE2_TEST_H
#define LIBTENSOR_BTO_SYMMETRIZE2_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::bto_symmetrize2 class

    \ingroup libtensor_tests_btod
 **/
template<typename T>
class bto_symmetrize2_test_x : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
    static const T k_thresh; //!< Threshold multiplier

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

class bto_symmetrize2_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTO_SYMMETRIZE2_TEST_H
