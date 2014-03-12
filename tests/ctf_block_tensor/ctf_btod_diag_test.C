#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_diag.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_diag.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include "../compare_ref.h"
#include "ctf_btod_diag_test.h"

namespace libtensor {


void ctf_btod_diag_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1a();
        test_1b();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_btod_diag_test::test_1a() {

    static const char testname[] = "ctf_btod_diag_test::test_1a()";

    typedef std_allocator<double> allocator_t;

    try {

    index<1> i1a, i1b;
    i1b[0] = 99;
    index<2> i2a, i2b;
    i2b[0] = 99; i2b[1] = 99;

    mask<1> m1;
    m1[0] = true;
    mask<2> m11;
    m11[0] = true; m11[1] = true;

    dimensions<2> dimsa(index_range<2>(i2a, i2b));
    dimensions<1> dimsb(index_range<1>(i1a, i1b));

    block_index_space<2> bisa(dimsa);
    bisa.split(m11, 40);
    block_index_space<1> bisb(dimsb);
    bisb.split(m1, 40);

    block_tensor<2, double, allocator_t> bta(bisa);
    block_tensor<1, double, allocator_t> btb(bisb), btb_ref(bisb);
    ctf_block_tensor<2, double> dbta(bisa);
    ctf_block_tensor<1, double> dbtb(bisb);

    btod_random<2>().perform(bta);
    btod_random<1>().perform(btb);
    btod_diag<2, 2>(bta, m11).perform(btb_ref);

    ctf_btod_distribute<2>(bta).perform(dbta);
    ctf_btod_distribute<1>(btb).perform(dbtb);

    ctf_btod_diag<2, 2>(dbta, m11).perform(dbtb);
    ctf_btod_collect<1>(dbtb).perform(btb);

    compare_ref<1>::compare(testname, btb, btb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_diag_test::test_1b() {

    static const char testname[] = "ctf_btod_diag_test::test_1b()";

    typedef std_allocator<double> allocator_t;

    try {

    index<1> i1a, i1b;
    i1b[0] = 99;
    index<2> i2a, i2b;
    i2b[0] = 99; i2b[1] = 99;

    mask<1> m1;
    m1[0] = true;
    mask<2> m11;
    m11[0] = true; m11[1] = true;

    dimensions<2> dimsa(index_range<2>(i2a, i2b));
    dimensions<1> dimsb(index_range<1>(i1a, i1b));

    block_index_space<2> bisa(dimsa);
    bisa.split(m11, 40);
    block_index_space<1> bisb(dimsb);
    bisb.split(m1, 40);

    block_tensor<2, double, allocator_t> bta(bisa);
    block_tensor<1, double, allocator_t> btb(bisb), btb_ref(bisb);
    ctf_block_tensor<2, double> dbta(bisa);
    ctf_block_tensor<1, double> dbtb(bisb);

    btod_random<2>().perform(bta);
    btod_random<1>().perform(btb);
    btod_copy<1>(btb).perform(btb_ref);
    btod_diag<2, 2>(bta, m11).perform(btb_ref, -0.5);

    ctf_btod_distribute<2>(bta).perform(dbta);
    ctf_btod_distribute<1>(btb).perform(dbtb);

    ctf_btod_diag<2, 2>(dbta, m11).perform(dbtb, -0.5);
    ctf_btod_collect<1>(dbtb).perform(btb);

    compare_ref<1>::compare(testname, btb, btb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

