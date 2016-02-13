#ifndef LIBTENSOR_CTF_EXPR_TEST_H
#define LIBTENSOR_CTF_EXPR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests various expressions

    \ingroup libtensor_ctf_iface_tests
 **/
class ctf_expr_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_conv_1();
    void test_conv_2();
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();
    void test_6();
    void test_7();

};

} // namespace libtensor

#endif // LIBTENSOR_CTF_EXPR_TEST_H
