#include "memory_reserve_test.h"
#include <libtensor/block_sparse/sparse_btensor_new.h>
#include <libtensor/block_sparse/memory_reserve.h>
#include "test_fixtures/contract2_subtract2_nested_test_f.h"

namespace libtensor {

void memory_reserve_test::perform() throw(libtest::test_exception)
{
    test_add_remove();
    test_add_not_enough_mem();
    test_remove_not_enough_tensors();
    test_tensor_destructor();
    test_memory_reserve_destructor();
    test_tensor_copy_constructor();
    test_reset_tensor_memory_reserve();
}

void memory_reserve_test::test_add_remove() throw(libtest::test_exception)
{
    static const char *test_name = "memory_reserve_test::test_add_remove()";

    memory_reserve mr(50);
    mr.add_tensor(18);

    if(mr.get_mem_avail() != 32)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "memory_reserve::get_mem_avail(...) did not return correct value");
    }
    if(mr.get_n_tensors() != 1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "memory_reserve::get_n_tensors(...) did not return correct value");
    }

    mr.remove_tensor(18);
    if(mr.get_mem_avail() != 50)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "memory_reserve::get_mem_avail(...) did not return correct value");
    }
    if(mr.get_n_tensors() != 0)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "memory_reserve::get_n_tensors(...) did not return correct value");
    }
}

void memory_reserve_test::test_add_not_enough_mem() throw(libtest::test_exception)
{
    static const char *test_name = "memory_reserve_test::test_add_not_enough_mem()";

    memory_reserve mr(50);
    bool threw_exception = false;
    try
    {
        mr.add_tensor(51);
        mr.remove_tensor(51);
    }
    catch(out_of_memory&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "memory_reserve::add_tensor(...) did not throw out of memory when tensor too large");
    }
}

void memory_reserve_test::test_remove_not_enough_tensors() throw(libtest::test_exception)
{
    static const char *test_name = "memory_reserve_test::test_remove_not_enough_tensors()";

    memory_reserve mr(50);
    bool threw_exception = false;
    try
    {
        mr.remove_tensor(18);
    }
    catch(generic_exception&)
    {
        threw_exception = true;
    }

    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "memory_reserve::remove_tensor(...) did not throw exception when no tensors present");
    }
}

void memory_reserve_test::test_tensor_destructor() throw(libtest::test_exception)
{
    static const char *test_name = "memory_reserve_test::test_tensor_destructor()";

    memory_reserve mr(402);
    //Scope here to force destructor call
    {
        sparse_bispace<1> foo_bispace(50);
        sparse_btensor<1> foo_tensor(foo_bispace);
        foo_tensor.set_memory_reserve(mr);
        if(mr.get_mem_avail() != 2 || mr.get_n_tensors() != 1)
        {
            fail_test(test_name,__FILE__,__LINE__,
                "sparse_btensor did not add self to memory reserve correctly");
        }
    }
    if(mr.get_mem_avail() != 402 || mr.get_n_tensors() != 0)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "sparse_btensor did not delete self from memory reserve correctly");
    }
}

void memory_reserve_test::test_memory_reserve_destructor() throw(libtest::test_exception)
{
    static const char *test_name = "memory_reserve_test::test_memory_reserve_destructor()";

    sparse_bispace<1> foo_bispace(50);
    sparse_btensor<1> foo_tensor(foo_bispace);
    bool threw_exception = false;
    try
    {
        memory_reserve mr(402);
        foo_tensor.set_memory_reserve(mr);
    }
    catch(generic_exception&)
    {
        threw_exception = true;
    }
    if(!threw_exception)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "memory_reserve did not threw exception when it was deleted with tensors still active");
    }
}

void memory_reserve_test::test_tensor_copy_constructor() throw(libtest::test_exception)
{
    static const char *test_name = "memory_reserve_test::test_tensor_copy_constructor()";

    sparse_bispace<1> foo_bispace(50);
    memory_reserve mr(900);
    sparse_btensor<1> foo_tensor(foo_bispace);
    foo_tensor.set_memory_reserve(mr);
    sparse_btensor<1> bar_tensor(foo_tensor);
    if(mr.get_mem_avail() != 100)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "memory reserve did not account properly for tensor copying");
    }
    if(mr.get_n_tensors() != 2)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "memory reserve did not account properly for tensor copying");
    }
    //TODO: NEED ASSIGNMENT OPERATOR TEST WHEN I HAVE ONE !!!
}

void memory_reserve_test::test_reset_tensor_memory_reserve() throw(libtest::test_exception)
{
    static const char *test_name = "memory_reserve_test::test_reset_tensor_memory_reserve()";

    sparse_bispace<1> foo_bispace(50);
    memory_reserve mr_0(400);
    memory_reserve mr_1(405);
    sparse_btensor<1> foo_tensor(foo_bispace);
    foo_tensor.set_memory_reserve(mr_0);
    foo_tensor.set_memory_reserve(mr_1);
    if(mr_0.get_mem_avail() != 400 || mr_0.get_n_tensors() != 0)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "Tensor did not properly remove from old memory reserve");
    }
    if(mr_1.get_mem_avail() != 5 || mr_1.get_n_tensors() != 1)
    {
        fail_test(test_name,__FILE__,__LINE__,
                "Tensor did not properly add to new memory reserve");
    }
}

} // namespace libtensor
