#ifndef SUBSPACE_ITERATOR_TEST_H
#define SUBSPACE_ITERATOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor
{

class subspace_iterator_test : public libtest::unit_test
{
public:
    virtual void perform() throw(libtest::test_exception);
private:
    void test_get_block_index_dense() throw(libtest::test_exception);
    void test_incr_dense() throw(libtest::test_exception);
    void test_get_block_index_sparse() throw(libtest::test_exception);
    void test_incr_sparse() throw(libtest::test_exception);
};

} /* namespace libtensor */


#endif /* SUBSPACE_ITERATOR_TEST_H */
