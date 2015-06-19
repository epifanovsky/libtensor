#ifndef BLOCK_LOOP_TEST_H
#define BLOCK_LOOP_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

class block_loop_test : public libtest::unit_test { 
public:
    virtual void perform() throw(libtest::test_exception);
private: 

    void test_contract2() throw(libtest::test_exception);
    void test_contract2_2d_2d_sparse_dense() throw(libtest::test_exception);
    void test_contract2_3d_2d_sparse_sparse() throw(libtest::test_exception);
};

} // namespace libtensor

#endif /* BLOCK_LOOP_TEST_H */
