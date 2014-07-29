#ifndef BATCH_PROVIDER_TEST_H
#define BATCH_PROVIDER_TEST_H

#include <libtest/unit_test.h>

namespace libtensor
{

class batch_provider_test: public libtest::unit_test
{
public:
    virtual void perform() throw(libtest::test_exception);
private:
    void test_permute_3d_sparse_120() throw(libtest::test_exception);
    void test_contract2() throw(libtest::test_exception);
    void test_contract2_permute_nested() throw(libtest::test_exception);
    void test_contract2_subtract2_nested() throw(libtest::test_exception);
    void test_batchable_subspaces_recursion_addition() throw(libtest::test_exception);
    void test_batchable_subspaces_recursion_permutation() throw(libtest::test_exception);
};

} /* namespace libtensor */


#endif /* BATCH_PROVIDER_TEST_H */
