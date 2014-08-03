#ifndef MEMORY_RESERVE_TEST_H
#define MEMORY_RESERVE_TEST_H

#include <libtest/unit_test.h>

namespace libtensor
{

class memory_reserve_test: public libtest::unit_test
{
public:
    virtual void perform() throw(libtest::test_exception);
private:
    void test_add_remove() throw(libtest::test_exception);
    void test_add_not_enough_mem() throw(libtest::test_exception);
    void test_remove_not_enough_tensors() throw(libtest::test_exception);
    void test_tensor_destructor() throw(libtest::test_exception);
    void test_memory_reserve_destructor() throw(libtest::test_exception);
    void test_tensor_copy_constructor() throw(libtest::test_exception);
    void test_reset_tensor_memory_reserve() throw(libtest::test_exception);
};

} /* namespace libtensor */


#endif /* MEMORY_RESERVE_TEST_H */
