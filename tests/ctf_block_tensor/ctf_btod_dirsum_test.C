#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_dirsum.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_dirsum.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include "../compare_ref.h"
#include "ctf_btod_dirsum_test.h"

namespace libtensor {


void ctf_btod_dirsum_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1a();
        test_1b();
        test_2a();
        test_2b();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_btod_dirsum_test::test_1a() {

    static const char testname[] = "ctf_btod_dirsum_test::test_1a()";

    typedef allocator<double> allocator_t;

    try {

    index<1> i1a, i1b;
    i1b[0] = 99;
    index<2> i2a, i2b;
    i2b[0] = 99; i2b[1] = 99;

    mask<1> m1;
    m1[0] = true;
    mask<2> m11;
    m11[0] = true; m11[1] = true;

    dimensions<1> dimsa(index_range<1>(i1a, i1b));
    dimensions<1> dimsb(index_range<1>(i1a, i1b));
    dimensions<2> dimsc(index_range<2>(i2a, i2b));

    block_index_space<1> bisa(dimsa);
    bisa.split(m1, 40);
    block_index_space<1> bisb(bisa);
    block_index_space<2> bisc(dimsc);
    bisc.split(m11, 40);

    block_tensor<1, double, allocator_t> bta(bisa);
    block_tensor<1, double, allocator_t> btb(bisb);
    block_tensor<2, double, allocator_t> btc(bisc), btc_ref(bisc);
    ctf_block_tensor<1, double> dbta(bisa);
    ctf_block_tensor<1, double> dbtb(bisb);
    ctf_block_tensor<2, double> dbtc(bisc);

    btod_random<1>().perform(bta);
    btod_random<1>().perform(btb);
    btod_random<2>().perform(btc);
    btod_dirsum<1, 1>(bta, 0.5, btb, -0.5).perform(btc_ref);

    ctf_btod_distribute<1>(bta).perform(dbta);
    ctf_btod_distribute<1>(btb).perform(dbtb);
    ctf_btod_distribute<2>(btc).perform(dbtc);

    ctf_btod_dirsum<1, 1>(dbta, 0.5, dbtb, -0.5).perform(dbtc);
    ctf_btod_collect<2>(dbtc).perform(btc);

    compare_ref<2>::compare(testname, btc, btc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_dirsum_test::test_1b() {

    static const char testname[] = "ctf_btod_dirsum_test::test_1b()";

    typedef allocator<double> allocator_t;

    try {

    index<1> i1a, i1b;
    i1b[0] = 99;
    index<2> i2a, i2b;
    i2b[0] = 99; i2b[1] = 99;

    mask<1> m1;
    m1[0] = true;
    mask<2> m11;
    m11[0] = true; m11[1] = true;

    dimensions<1> dimsa(index_range<1>(i1a, i1b));
    dimensions<1> dimsb(index_range<1>(i1a, i1b));
    dimensions<2> dimsc(index_range<2>(i2a, i2b));

    block_index_space<1> bisa(dimsa);
    bisa.split(m1, 40);
    block_index_space<1> bisb(bisa);
    block_index_space<2> bisc(dimsc);
    bisc.split(m11, 40);

    block_tensor<1, double, allocator_t> bta(bisa);
    block_tensor<1, double, allocator_t> btb(bisb);
    block_tensor<2, double, allocator_t> btc(bisc), btc_ref(bisc);
    ctf_block_tensor<1, double> dbta(bisa);
    ctf_block_tensor<1, double> dbtb(bisb);
    ctf_block_tensor<2, double> dbtc(bisc);

    btod_random<1>().perform(bta);
    btod_random<1>().perform(btb);
    btod_random<2>().perform(btc);
    btod_copy<2>(btc).perform(btc_ref);
    btod_dirsum<1, 1>(bta, 0.5, btb, -0.5).perform(btc_ref, 2.0);

    ctf_btod_distribute<1>(bta).perform(dbta);
    ctf_btod_distribute<1>(btb).perform(dbtb);
    ctf_btod_distribute<2>(btc).perform(dbtc);

    ctf_btod_dirsum<1, 1>(dbta, 0.5, dbtb, -0.5).perform(dbtc, 2.0);
    ctf_btod_collect<2>(dbtc).perform(btc);

    compare_ref<2>::compare(testname, btc, btc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_dirsum_test::test_2a() {

    static const char testname[] = "ctf_btod_dirsum_test::test_2a()";

    typedef allocator<double> allocator_t;

    try {

    index<1> i1a, i1b;
    i1b[0] = 99;
    index<3> i3a, i3b;
    i3b[0] = 19; i3b[1] = 9; i3b[2] = 29;
    index<4> i4a, i4b;
    i4b[0] = 9; i4b[1] = 19; i4b[2] = 29; i4b[3] = 99;

    mask<1> m1;
    m1[0] = true;
    mask<3> m100, m001;
    m100[0] = true; m001[2] = true;
    mask<4> m0100, m0010, m0001;
    m0100[1] = true; m0010[2] = true; m0001[3] = true;

    dimensions<1> dimsa(index_range<1>(i1a, i1b));
    dimensions<3> dimsb(index_range<3>(i3a, i3b));
    dimensions<4> dimsc(index_range<4>(i4a, i4b));

    block_index_space<1> bisa(dimsa);
    bisa.split(m1, 40);
    block_index_space<3> bisb(dimsb);
    bisb.split(m100, 10);
    bisb.split(m001, 12);
    block_index_space<4> bisc(dimsc);
    bisc.split(m0100, 10);
    bisc.split(m0010, 12);
    bisc.split(m0001, 40);

    block_tensor<1, double, allocator_t> bta(bisa);
    block_tensor<3, double, allocator_t> btb(bisb);
    block_tensor<4, double, allocator_t> btc(bisc), btc_ref(bisc);
    ctf_block_tensor<1, double> dbta(bisa);
    ctf_block_tensor<3, double> dbtb(bisb);
    ctf_block_tensor<4, double> dbtc(bisc);

    permutation<4> permc; // ljik -> ijkl
    permc.permute(0, 2).permute(2, 3);

    btod_random<1>().perform(bta);
    btod_random<3>().perform(btb);
    btod_random<4>().perform(btc);
    btod_dirsum<1, 3>(bta, 0.5, btb, -0.5, permc).perform(btc_ref);

    ctf_btod_distribute<1>(bta).perform(dbta);
    ctf_btod_distribute<3>(btb).perform(dbtb);
    ctf_btod_distribute<4>(btc).perform(dbtc);

    ctf_btod_dirsum<1, 3>(dbta, 0.5, dbtb, -0.5, permc).perform(dbtc);
    ctf_btod_collect<4>(dbtc).perform(btc);

    compare_ref<4>::compare(testname, btc, btc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_dirsum_test::test_2b() {

    static const char testname[] = "ctf_btod_dirsum_test::test_2b()";

    typedef allocator<double> allocator_t;

    try {

    index<1> i1a, i1b;
    i1b[0] = 99;
    index<3> i3a, i3b;
    i3b[0] = 19; i3b[1] = 9; i3b[2] = 29;
    index<4> i4a, i4b;
    i4b[0] = 9; i4b[1] = 19; i4b[2] = 29; i4b[3] = 99;

    mask<1> m1;
    m1[0] = true;
    mask<3> m100, m001;
    m100[0] = true; m001[2] = true;
    mask<4> m0100, m0010, m0001;
    m0100[1] = true; m0010[2] = true; m0001[3] = true;

    dimensions<1> dimsa(index_range<1>(i1a, i1b));
    dimensions<3> dimsb(index_range<3>(i3a, i3b));
    dimensions<4> dimsc(index_range<4>(i4a, i4b));

    block_index_space<1> bisa(dimsa);
    bisa.split(m1, 40);
    block_index_space<3> bisb(dimsb);
    bisb.split(m100, 10);
    bisb.split(m001, 12);
    block_index_space<4> bisc(dimsc);
    bisc.split(m0100, 10);
    bisc.split(m0010, 12);
    bisc.split(m0001, 40);

    block_tensor<1, double, allocator_t> bta(bisa);
    block_tensor<3, double, allocator_t> btb(bisb);
    block_tensor<4, double, allocator_t> btc(bisc), btc_ref(bisc);
    ctf_block_tensor<1, double> dbta(bisa);
    ctf_block_tensor<3, double> dbtb(bisb);
    ctf_block_tensor<4, double> dbtc(bisc);

    permutation<4> permc; // ljik -> ijkl
    permc.permute(0, 2).permute(2, 3);

    btod_random<1>().perform(bta);
    btod_random<3>().perform(btb);
    btod_random<4>().perform(btc);
    btod_copy<4>(btc).perform(btc_ref);
    btod_dirsum<1, 3>(bta, 0.5, btb, -0.5, permc).perform(btc_ref, 2.0);

    ctf_btod_distribute<1>(bta).perform(dbta);
    ctf_btod_distribute<3>(btb).perform(dbtb);
    ctf_btod_distribute<4>(btc).perform(dbtc);

    ctf_btod_dirsum<1, 3>(dbta, 0.5, dbtb, -0.5, permc).perform(dbtc, 2.0);
    ctf_btod_collect<4>(dbtc).perform(btc);

    compare_ref<4>::compare(testname, btc, btc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

