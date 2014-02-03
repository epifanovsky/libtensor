#ifndef DIRECT_SPARSE_BTENSOR_TEST_H
#define DIRECT_SPARSE_BTENSOR_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

class direct_sparse_btensor_test : public libtest::unit_test { 
public:
    virtual void perform() throw(libtest::test_exception);
private:
    void test_get_batch_contract2() throw(libtest::test_exception);
    void test_contract2_direct_rhs() throw(libtest::test_exception);
};

} // namespace libtensor

#endif /* DIRECT_SPARSE_BTENSOR_TEST_H */
