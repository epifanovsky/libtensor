#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include "block_tensor_test.h"

namespace libtensor {

void block_tensor_test::perform() throw(libtest::test_exception) {

    test_req_aux_block_1();
    test_orbits_1();
    test_orbits_2();
    test_orbits_3();
}


void block_tensor_test::test_req_aux_block_1() throw(libtest::test_exception) {

    static const char *testname =
        "block_tensor_test::test_req_aux_block_1()";

    typedef std_allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 4; i2[1] = 4;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m; m[0] = true; m[1] = true;
    bis.split(m, 2);

    block_tensor<2, double, allocator_t> bt(bis);
    block_tensor_ctrl<2, double> ctrl(bt);

    index<2> i00, i01, i10, i11;
    i10[0] = 1; i01[1] = 1;
    i11[0] = 1; i11[1] = 1;

    dense_tensor_i<2, double> &b00 = ctrl.req_aux_block(i00);
    dense_tensor_i<2, double> &b01 = ctrl.req_aux_block(i01);
    dense_tensor_i<2, double> &b10 = ctrl.req_aux_block(i10);
    dense_tensor_i<2, double> &b11 = ctrl.req_aux_block(i11);

    tod_random<2>().perform(b00);
    tod_random<2>().perform(b01);
    tod_random<2>().perform(b10);
    tod_random<2>().perform(b11);

    ctrl.ret_aux_block(i00);
    ctrl.ret_aux_block(i01);
    ctrl.ret_aux_block(i10);
    ctrl.ret_aux_block(i11);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void block_tensor_test::test_orbits_1() throw(libtest::test_exception) {
/*
    static const char *testname = "block_tensor_test::test_orbits_1()";

    typedef std_allocator<double> allocator_t;
    typedef block_tensor<2, double, allocator_t> block_tensor_type;
    typedef block_tensor_ctrl<2, double> block_tensor_ctrl_t;
    typedef orbit_iterator<2, double> orbit_iterator_t;

    try {

    index<2> i0, i1, i2;
    i2[0] = 4; i2[1] = 4;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    block_tensor_type bt(bis);
    block_tensor_ctrl_t ctrl(bt);
    orbit_iterator_t oi = ctrl.req_orbits();

    if(!oi.end()) {
        fail_test(testname, __FILE__, __LINE__,
            "Expecting an empty block set for a new block tensor.");
    }

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }*/
}

void block_tensor_test::test_orbits_2() throw(libtest::test_exception) {
/*
    static const char *testname = "block_tensor_test::test_orbits_2()";

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

void block_tensor_test::test_orbits_3() throw(libtest::test_exception) {
/*
    static const char *testname = "block_tensor_test::test_orbits_3()";

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
