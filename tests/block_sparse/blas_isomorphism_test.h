#ifndef BLAS_ISOMORPHISM_TEST_H
#define BLAS_ISOMORPHISM_TEST_H

#include <libtest/unit_test.h>

namespace libtensor
{

class blas_isomorphism_test : public libtest::unit_test
{
public:
    virtual void perform() throw(libtest::test_exception);
private:
    void test_matmul_isomorphism_params_identity_NN() throw(libtest::test_exception);
    void test_matmul_isomorphism_params_identity_NT() throw(libtest::test_exception);
    void test_matmul_isomorphism_params_identity_TN() throw(libtest::test_exception);
    void test_matmul_isomorphism_params_identity_TT() throw(libtest::test_exception);
};

} /* namespace libtensor */


#endif /* BLAS_ISOMORPHISM_TEST_H */
