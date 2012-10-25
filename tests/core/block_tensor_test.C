#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include "block_tensor_test.h"

namespace libtensor {


void block_tensor_test::perform() throw(libtest::test_exception) {

    test_nonzero_blocks_1();
    test_nonzero_blocks_2();
    test_nonzero_blocks_3();
}


void block_tensor_test::test_nonzero_blocks_1() {

    static const char *testname = "block_tensor_test::test_nonzero_blocks_1()";

    typedef std_allocator<double> allocator_type;
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
/*
    static const char *testname = "block_tensor_test::test_nonzero_blocks_2()";

    typedef std_allocator<double> allocator_t;
    typedef dense_tensor_i<2, double> block_type;
    typedef block_tensor<2, double, allocator_t> block_tensor_type;
    typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;
    typedef orbit_iterator<2, double> orbit_iterator_t;
    typedef std::map< index<2>, bool > map_t;

    try {

    index<2> i0, i1, i2;
    i2[0] = 4; i2[1] = 4;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(0, 2);
    bis.split(1, 2);

    block_tensor_type bt(bis);
    block_tensor_ctrl_t ctrl(bt);

    index<2> ii;
    block_type &blk_00 = ctrl.req_block(ii);
    ctrl.ret_block(ii);
    map_t map;
    map[i0] = false;

    orbit_iterator_t oi = ctrl.req_orbits();
    size_t total = 0;

    while(!oi.end()) {
        map_t::iterator iter = map.find(oi.get_index());
        if(iter == map.end()) {
            fail_test(testname, __FILE__, __LINE__,
                "Unexpected orbit index.");
        }
        if(iter->second == true) {
            fail_test(testname, __FILE__, __LINE__,
                "Repeated orbit index.");
        }
        iter->second = true;
        oi.next();
        total++;
    }

    if(map.size() != total) {
        fail_test(testname, __FILE__, __LINE__,
            "Set of orbits is incomplete.");
    }

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }*/
}


void block_tensor_test::test_nonzero_blocks_3() {
/*
    static const char *testname = "block_tensor_test::test_nonzero_blocks_3()";

    typedef std_allocator<double> allocator_t;
    typedef dense_tensor_i<2, double> block_type;
    typedef block_tensor<2, double, allocator_t> block_tensor_type;
    typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;
    typedef orbit_iterator<2, double> orbit_iterator_t;
    typedef std::map< index<2>, bool > map_t;

    try {

    index<2> i0, i1, i2;
    i2[0] = 4; i2[1] = 4;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    bis.split(0, 2);
    bis.split(1, 2);

    block_tensor_type bt(bis);
    block_tensor_ctrl_t ctrl(bt);

    map_t map;
    index<2> ii;
    block_type &blk_00 = ctrl.req_block(ii);
    map[ii] = false;
    ctrl.ret_block(ii);
    ii[0] = 1; ii[1] = 1;
    block_type &blk_11 = ctrl.req_block(ii);
    map[ii] = false;
    ctrl.ret_block(ii);

    orbit_iterator_t oi = ctrl.req_orbits();
    size_t total = 0;

    while(!oi.end()) {
        map_t::iterator iter = map.find(oi.get_index());
        if(iter == map.end()) {
            fail_test(testname, __FILE__, __LINE__,
                "Unexpected orbit index.");
        }
        if(iter->second == true) {
            fail_test(testname, __FILE__, __LINE__,
                "Repeated orbit index.");
        }
        iter->second = true;
        oi.next();
        total++;
    }

    if(map.size() != total) {
        fail_test(testname, __FILE__, __LINE__,
            "Set of orbits is incomplete.");
    }

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }*/
}


} // namespace libtensor
