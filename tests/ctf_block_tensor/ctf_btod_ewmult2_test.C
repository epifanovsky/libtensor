#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/block_tensor/btod_ewmult2.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include <libtensor/ctf_block_tensor/ctf_btod_ewmult2.h>
#include "../compare_ref.h"
#include "ctf_btod_ewmult2_test.h"

namespace libtensor {


void ctf_btod_ewmult2_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_btod_ewmult2_test::test_1() {

    static const char testname[] = "ctf_btod_ewmult2_test::test_1()";

    typedef allocator<double> allocator_t;

    try {

    mask<1> m1;
    m1[0] = true;
    mask<3> m101;
    m101[0] = true; m101[2] = true;

    index<1> i1a, i1b;
    i1b[0] = 4;
    index<3> i3a, i3b;
    i3b[0] = 99; i3b[1] = 4; i3b[2] = 99;
    block_index_space<3> bisa(dimensions<3>(index_range<3>(i3a, i3b)));
    bisa.split(m101, 30);
    block_index_space<1> bisb(dimensions<1>(index_range<1>(i1a, i1b)));
    block_index_space<3> bisc(bisa);

    block_tensor<3, double, allocator_t> bta(bisa);
    block_tensor<1, double, allocator_t> btb(bisb);
    block_tensor<3, double, allocator_t> btc(bisc), btc_ref(bisc);
    ctf_block_tensor<3, double> dbta(bisa);
    ctf_block_tensor<1, double> dbtb(bisb);
    ctf_block_tensor<3, double> dbtc(bisc);

    btod_random<3>().perform(bta);
    btod_random<1>().perform(btb);
    btod_random<3>().perform(btc);

    ctf_btod_distribute<3>(bta).perform(dbta);
    ctf_btod_distribute<1>(btb).perform(dbtb);
    ctf_btod_distribute<3>(btc).perform(dbtc);

    permutation<3> perma;
    perma.permute(1, 2);
    permutation<1> permb;
    permutation<3> permc;
    permc.permute(1, 2).permute(0, 2);
    btod_ewmult2<2, 0, 1>(bta, perma, btb, permb, permc).perform(btc_ref);

    ctf_btod_ewmult2<2, 0, 1>(dbta, perma, dbtb, permb, permc).perform(dbtc);
    ctf_btod_collect<3>(dbtc).perform(btc);

    compare_ref<3>::compare(testname, btc, btc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

