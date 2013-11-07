#ifndef LOOP_LIST_SPARSITY_DATA_TEST_H
#define LOOP_LIST_SPARSITY_DATA_TEST_H

#include <libtest/unit_test.h>

namespace libtensor {

class loop_list_sparsity_data_test : public libtest::unit_test { 
public:
    virtual void perform() throw(libtest::test_exception);
private:
    //Order of loops matches order of tree in bispace
    void test_get_sig_block_list_in_order() throw(libtest::test_exception);
    //Must fuse input and output sparsity to get correct answer
    //Requires permuting sparsity of second input tensor
    void test_get_sig_block_list_fuse_output_input() throw(libtest::test_exception);
};

} // namespace libtensor

#endif /* LOOP_LIST_SPARSITY_DATA_TEST_H */
