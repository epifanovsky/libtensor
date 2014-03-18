#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/block_tensor/btod_mult.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include <libtensor/ctf_block_tensor/ctf_btod_mult.h>
#include "../compare_ref.h"
#include "ctf_btod_mult_test.h"

namespace libtensor {


void ctf_btod_mult_test::perform() throw(libtest::test_exception) {

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


void ctf_btod_mult_test::test_mult_1() {

    static const char testname[] = "ctf_btod_mult_test::test_mult_1()";

    typedef std_allocator<double> allocator_t;

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
    block_index_space<3> bisc(bisa);

    block_tensor<3, double, allocator_t> bta(bisa), btb(bisb), btc(bisc),
        btc_ref(bisc);
    ctf_block_tensor<3, double> dbta(bisa), dbtb(bisb), dbtc(bisc);

    btod_random<3>().perform(bta);
    btod_random<3>().perform(btb);
    btod_random<3>().perform(btc);

    ctf_btod_distribute<3>(bta).perform(dbta);
    ctf_btod_distribute<3>(btb).perform(dbtb);
    ctf_btod_distribute<3>(btc).perform(dbtc);

    btod_mult<3>(bta, p012, btb, p201, false, 0.5).perform(btc_ref);

    ctf_btod_mult<3>(dbta, p012, dbtb, p201, false, 0.5).perform(dbtc);
    ctf_btod_collect<3>(dbtc).perform(btc);

    compare_ref<3>::compare(testname, btc, btc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_mult_test::test_div_1() {

    static const char testname[] = "ctf_btod_mult_test::test_div_1()";

    typedef std_allocator<double> allocator_t;

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
    block_index_space<3> bisc(bisa);

    block_tensor<3, double, allocator_t> bta(bisa), btb(bisb), btc(bisc),
        btc_ref(bisc);
    ctf_block_tensor<3, double> dbta(bisa), dbtb(bisb), dbtc(bisc);

    btod_random<3>().perform(bta);
    btod_random<3>().perform(btb);
    btod_random<3>().perform(btc);

    ctf_btod_distribute<3>(bta).perform(dbta);
    ctf_btod_distribute<3>(btb).perform(dbtb);
    ctf_btod_distribute<3>(btc).perform(dbtc);

    btod_mult<3>(bta, p012, btb, p201, true, 0.5).perform(btc_ref);

    ctf_btod_mult<3>(dbta, p012, dbtb, p201, true, 0.5).perform(dbtc);
    ctf_btod_collect<3>(dbtc).perform(btc);

    compare_ref<3>::compare(testname, btc, btc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

