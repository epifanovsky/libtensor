#ifndef LIBTENSOR_EXPR_TREE_TEST_H
#define LIBTENSOR_EXPR_TREE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

/** \brief Tests class iface::expr_tree

    \ingroup libtensor_tests_iface
 **/
class expr_tree_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_1();
    void test_2();
    void test_3();
    void test_4();
    void test_5();
    void test_6();
    void test_7();

};

} // namespace libtensor

#endif // LIBTENSOR_EXPR_TREE_TEST_H
