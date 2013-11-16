#ifndef LOOP_LIST_SPARSITY_DATA_TEST_H
#define LOOP_LIST_SPARSITY_DATA_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

class loop_list_sparsity_data_test : public libtest::unit_test { 
public:
    virtual void perform() throw(libtest::test_exception);
private:
    //No sparsity at all: simplest case
    void test_get_sig_block_list_no_sparsity() throw(libtest::test_exception);

    //Sparsity in a single tensor in the expression, as when performing a load
    void test_get_sig_block_list_sparsity_one_tensor() throw(libtest::test_exception);

    //Must fuse input and output sparsity to get correct answer
    //Requires permuting sparsity of second input tensor
    void test_get_sig_block_list_sparsity_3_tensors() throw(libtest::test_exception);


    //Fuse 3 tensors all at once - first two tensors do not couple by themselves
    //but third tensor can couple them
    void test_get_sig_block_list_sparsity_delayed_fuse() throw(libtest::test_exception);
};

} // namespace libtensor

#endif /* LOOP_LIST_SPARSITY_DATA_TEST_H */
