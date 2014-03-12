#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/block_tensor/btod_contract2.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_contract2.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include "../compare_ref.h"
#include "ctf_btod_contract2_test.h"

namespace libtensor {


void ctf_btod_contract2_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_btod_contract2_test::test_1() {

    static const char testname[] = "ctf_btod_contract2_test::test_1()";

    typedef std_allocator<double> allocator_t;

    try {

    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2));
    block_index_space<2> bisa(dimsa);
    bisa.split(m11, 30);
    block_index_space<2> bisb(bisa), bisc(bisa);

    block_tensor<2, double, allocator_t> bta(bisa), btb(bisb), btc(bisc),
        btc_ref(bisc);
    ctf_block_tensor<2, double> dbta(bisa), dbtb(bisb), dbtc(bisc);

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);
    btod_random<2>().perform(btc);

    ctf_btod_distribute<2>(bta).perform(dbta);
    ctf_btod_distribute<2>(btb).perform(dbtb);
    ctf_btod_distribute<2>(btc).perform(dbtc);

    contraction2<1, 1, 1> contr;
    contr.contract(1, 0);

    btod_contract2<1, 1, 1>(contr, bta, btb).perform(btc_ref);

    ctf_btod_contract2<1, 1, 1>(contr, dbta, dbtb).perform(dbtc);
    ctf_btod_collect<2>(dbtc).perform(btc);

    compare_ref<2>::compare(testname, btc, btc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

