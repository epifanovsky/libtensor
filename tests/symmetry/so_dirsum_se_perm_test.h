#ifndef LIBTENSOR_SO_DIRSUM_SE_PERM_TEST_H
#define LIBTENSOR_SO_DIRSUM_SE_PERM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the class
        symmetry_operation_impl< so_dirsum<N, M, T>, se_perm<N + M, T> >

	\ingroup libtensor_tests_sym
 **/
class so_dirsum_se_perm_test : public libtest::unit_test {
public:
    virtual void perform() throw(libtest::test_exception);

private:
    void test_empty_1() throw(libtest::test_exception);
    void test_empty_2(bool perm) throw(libtest::test_exception);
    void test_empty_3(bool perm) throw(libtest::test_exception);
    void test_nn_1(bool symm1, bool symm2) throw(libtest::test_exception);
    void test_nn_2(bool symm1, bool symm2) throw(libtest::test_exception);

};


} // namespace libtensor

#endif // LIBTENSOR_SO_DIRSUM_SE_PERM_TEST_H

