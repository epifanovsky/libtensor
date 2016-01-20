#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_mult1.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include <libtensor/ctf_block_tensor/ctf_btod_mult1.h>
#include "../compare_ref.h"
#include "ctf_btod_mult1_test.h"

namespace libtensor {


void ctf_btod_mult1_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_mult_1();
        test_div_1();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_btod_mult1_test::test_mult_1() {

    static const char testname[] = "ctf_btod_mult1_test::test_mult_1()";

    typedef allocator<double> allocator_t;

    try {

    mask<3> m101;
    m101[0] = true; m101[2] = true;

    permutation<3> p012, p120, p201;
    p120.permute(0, 2).permute(0, 1);
    p201.permute(0, 1).permute(0, 2);

    index<3> i3a, i3b;
    i3b[0] = 49; i3b[1] = 9; i3b[2] = 49;
    block_index_space<3> bisa(dimensions<3>(index_range<3>(i3a, i3b)));
    bisa.split(m101, 30);
    block_index_space<3> bisb(bisa);
    bisb.permute(p120);

    block_tensor<3, double, allocator_t> bta(bisa), bta_ref(bisa), btb(bisb);
    ctf_block_tensor<3, double> dbta(bisa), dbtb(bisb);

    btod_random<3>().perform(bta);
    btod_random<3>().perform(btb);

    ctf_btod_distribute<3>(bta).perform(dbta);
    ctf_btod_distribute<3>(btb).perform(dbtb);

    btod_copy<3>(bta).perform(bta_ref);
    btod_mult1<3>(btb, p201, false, 0.5).perform(false, bta_ref);

    ctf_btod_mult1<3>(dbtb, p201, false, 0.5).perform(false, dbta);
    ctf_btod_collect<3>(dbta).perform(bta);

    compare_ref<3>::compare(testname, bta, bta_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_mult1_test::test_div_1() {

    static const char testname[] = "ctf_btod_mult1_test::test_div_1()";

    typedef allocator<double> allocator_t;

    try {

    mask<3> m101;
    m101[0] = true; m101[2] = true;

    permutation<3> p012, p120, p201;
    p120.permute(0, 2).permute(0, 1);
    p201.permute(0, 1).permute(0, 2);

    index<3> i3a, i3b;
    i3b[0] = 49; i3b[1] = 9; i3b[2] = 49;
    block_index_space<3> bisa(dimensions<3>(index_range<3>(i3a, i3b)));
    bisa.split(m101, 30);
    block_index_space<3> bisb(bisa);
    bisb.permute(p120);

    block_tensor<3, double, allocator_t> bta(bisa), bta_ref(bisa), btb(bisb);
    ctf_block_tensor<3, double> dbta(bisa), dbtb(bisb);

    btod_random<3>().perform(bta);
    btod_random<3>().perform(btb);

    ctf_btod_distribute<3>(bta).perform(dbta);
    ctf_btod_distribute<3>(btb).perform(dbtb);

    btod_copy<3>(bta).perform(bta_ref);
    btod_mult1<3>(btb, p201, true, 0.5).perform(false, bta_ref);

    ctf_btod_mult1<3>(dbtb, p201, true, 0.5).perform(false, dbta);
    ctf_btod_collect<3>(dbta).perform(bta);

    compare_ref<3>::compare(testname, bta, bta_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

