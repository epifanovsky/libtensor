#ifndef LIBTENSOR_SO_DIRPROD_SE_PART_TEST_H
#define LIBTENSOR_SO_DIRPROD_SE_PART_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {


/**	\brief Tests the class libtensor::
        symmetry_operation_impl<so_dirprod<N, M, T>, se_part<N + M, T> >

	\ingroup libtensor_tests_sym
 **/
class so_dirprod_se_part_test : public libtest::unit_test {
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

#endif // LIBTENSOR_SO_DIRPROD_SE_PART_TEST_H

