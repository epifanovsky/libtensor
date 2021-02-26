#ifndef LIBTENSOR_BTO_COPY_TEST_H
#define LIBTENSOR_BTO_COPY_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/** \brief Tests the libtensor::bto_copy class

    \ingroup libtensor_tests_btod
 **/
template<typename T>
class bto_copy_test_x : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
    static const T k_thresh;

private:
    void test_zero_1() throw(libtest::test_exception);
    void test_zero_2() throw(libtest::test_exception);
    void test_nosym_1() throw(libtest::test_exception);
    void test_nosym_2() throw(libtest::test_exception);
    void test_nosym_3() throw(libtest::test_exception);
    void test_nosym_4() throw(libtest::test_exception);
    void test_sym_1() throw(libtest::test_exception);
    void test_sym_2() throw(libtest::test_exception);
    void test_sym_3() throw(libtest::test_exception);
    void test_sym_4() throw(libtest::test_exception);
    void test_add_nosym_1() throw(libtest::test_exception);
    void test_add_nosym_2() throw(libtest::test_exception);
    void test_add_nosym_3() throw(libtest::test_exception);
    void test_add_nosym_4() throw(libtest::test_exception);
    void test_add_eqsym_1() throw(libtest::test_exception);
    void test_add_eqsym_2() throw(libtest::test_exception);
    void test_add_eqsym_3() throw(libtest::test_exception);
    void test_add_eqsym_4() throw(libtest::test_exception);
    void test_add_eqsym_5() throw(libtest::test_exception);
    void test_add_nesym_1() throw(libtest::test_exception);
    void test_add_nesym_2() throw(libtest::test_exception);
    void test_add_nesym_3() throw(libtest::test_exception);
    void test_add_nesym_4() throw(libtest::test_exception);
    void test_add_nesym_5() throw(libtest::test_exception);
    void test_add_nesym_5_sp() throw(libtest::test_exception);
    void test_add_nesym_6() throw(libtest::test_exception);
    void test_add_nesym_7_sp1() throw(libtest::test_exception);
    void test_add_nesym_7_sp2() throw(libtest::test_exception);
    void test_add_nesym_7_sp3() throw(libtest::test_exception);

    void test_dir_1() throw(libtest::test_exception);
    void test_dir_2() throw(libtest::test_exception);
    void test_dir_3() throw(libtest::test_exception);
    void test_dir_4() throw(libtest::test_exception);

    void test_bug_1() throw(libtest::test_exception);

};

class bto_copy_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);
};

} // namespace libtensor

#endif // LIBTENSOR_BTO_COPY_TEST_H
