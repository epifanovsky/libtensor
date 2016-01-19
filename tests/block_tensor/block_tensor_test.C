#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include "block_tensor_test.h"

namespace libtensor {


void block_tensor_test::perform() throw(libtest::test_exception) {

    test_nonzero_blocks_1();
    test_nonzero_blocks_2();
}


void block_tensor_test::test_nonzero_blocks_1() {

    static const char *testname = "block_tensor_test::test_nonzero_blocks_1()";

    typedef allocator<double> allocator_type;
    typedef block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i0, i1, i2;
    i2[0] = 4; i2[1] = 4;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    block_tensor<2, double, allocator_type> bta(bis);
    gen_block_tensor_rd_ctrl<2, bti_traits> ca(bta);

    std::vector<size_t> nzl1, nzl2;
    nzl2.push_back(100);
    nzl2.push_back(200);
    std::vector<size_t> nzl3(nzl1), nzl4(nzl2);

    ca.req_nonzero_blocks(nzl1);
    if(!nzl1.empty()) {
        fail_test(testname, __FILE__, __LINE__, "!nzl1.empty()");
    }

    ca.req_nonzero_blocks(nzl2);
    if(!nzl2.empty()) {
        fail_test(testname, __FILE__, __LINE__, "!nzl2.empty()");
    }

    {
        index<2> i00;
        gen_block_tensor_wr_ctrl<2, bti_traits> ca1(bta);
        dense_tensor_wr_i<2, double> &b00 = ca1.req_block(i00);
        tod_random<2>().perform(b00);
        ca1.ret_block(i00);
    }

    ca.req_nonzero_blocks(nzl3);
    if(nzl3.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "nzl3.size() != 1");
    }
    if(nzl3[0] != 0) {
        fail_test(testname, __FILE__, __LINE__, "nzl3[0] != 0");
    }

    ca.req_nonzero_blocks(nzl4);
    if(nzl4.size() != 1) {
        fail_test(testname, __FILE__, __LINE__, "nzl4.size() != 1");
    }
    if(nzl4[0] != 0) {
        fail_test(testname, __FILE__, __LINE__, "nzl4[0] != 0");
    }

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void block_tensor_test::test_nonzero_blocks_2() {

    static const char *testname = "block_tensor_test::test_nonzero_blocks_2()";

    typedef allocator<double> allocator_type;
    typedef block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i0, i1, i2;
    i2[0] = 4; i2[1] = 4;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    bis.split(m11, 3);

    block_tensor<2, double, allocator_type> bta(bis);
    gen_block_tensor_rd_ctrl<2, bti_traits> ca(bta);

    std::vector<size_t> nzl1, nzl2;
    nzl2.push_back(100);
    nzl2.push_back(200);
    std::vector<size_t> nzl3(nzl1), nzl4(nzl2);

    ca.req_nonzero_blocks(nzl1);
    if(!nzl1.empty()) {
        fail_test(testname, __FILE__, __LINE__, "!nzl1.empty()");
    }

    ca.req_nonzero_blocks(nzl2);
    if(!nzl2.empty()) {
        fail_test(testname, __FILE__, __LINE__, "!nzl2.empty()");
    }

    {
        index<2> i00, i10;
        i10[0] = 1; i10[1] = 0;
        gen_block_tensor_wr_ctrl<2, bti_traits> ca1(bta);
        dense_tensor_wr_i<2, double> &b00 = ca1.req_block(i00);
        tod_random<2>().perform(b00);
        ca1.ret_block(i00);
        dense_tensor_wr_i<2, double> &b10 = ca1.req_block(i10);
        tod_random<2>().perform(b10);
        ca1.ret_block(i10);
    }

    ca.req_nonzero_blocks(nzl3);
    if(nzl3.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "nzl3.size() != 2");
    }
    if(std::find(nzl3.begin(), nzl3.end(), 0) == nzl3.end()) {
        fail_test(testname, __FILE__, __LINE__, "nzl3 doesn't contain [0,0]");
    }
    if(std::find(nzl3.begin(), nzl3.end(), 2) == nzl3.end()) {
        fail_test(testname, __FILE__, __LINE__, "nzl3 doesn't contain [1,0]");
    }

    ca.req_nonzero_blocks(nzl4);
    if(nzl4.size() != 2) {
        fail_test(testname, __FILE__, __LINE__, "nzl4.size() != 2");
    }
    if(std::find(nzl4.begin(), nzl4.end(), 0) == nzl4.end()) {
        fail_test(testname, __FILE__, __LINE__, "nzl4 doesn't contain [0,0]");
    }
    if(std::find(nzl4.begin(), nzl4.end(), 2) == nzl4.end()) {
        fail_test(testname, __FILE__, __LINE__, "nzl4 doesn't contain [1,0]");
    }

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


} // namespace libtensor
