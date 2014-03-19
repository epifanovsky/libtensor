#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/block_tensor/btod_scale.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include <libtensor/ctf_block_tensor/ctf_btod_scale.h>
#include "../compare_ref.h"
#include "ctf_btod_scale_test.h"

namespace libtensor {


void ctf_btod_scale_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1();
        test_2();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_btod_scale_test::test_1() {

    static const char testname[] = "ctf_btod_scale_test::test_1()";

    typedef std_allocator<double> allocator_t;

    try {

    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2));
    block_index_space<2> bisa(dimsa);
    bisa.split(m11, 30);

    block_tensor<2, double, allocator_t> bta(bisa), bta_ref(bisa);
    ctf_block_tensor<2, double> dbta(bisa);

    btod_random<2>().perform(bta);
    btod_copy<2>(bta).perform(bta_ref);
    btod_scale<2>(bta_ref, -0.25).perform();

    ctf_btod_distribute<2>(bta).perform(dbta);
    ctf_btod_scale<2>(dbta, -0.25).perform();
    ctf_btod_collect<2>(dbta).perform(bta);

    compare_ref<2>::compare(testname, bta, bta_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_scale_test::test_2() {

    static const char testname[] = "ctf_btod_scale_test::test_2()";

    typedef std_allocator<double> allocator_t;

    try {

    mask<4> m1111;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

    index<4> i1, i2;
    i2[0] = 19; i2[1] = 19; i2[2] = 19; i2[3] = 19;
    dimensions<4> dimsa(index_range<4>(i1, i2));
    block_index_space<4> bisa(dimsa);
    bisa.split(m1111, 9);

    block_tensor<4, double, allocator_t> bta(bisa), bta_ref(bisa);
    ctf_block_tensor<4, double> dbta(bisa);

    btod_random<4>().perform(bta);
    btod_copy<4>(bta).perform(bta_ref);
    btod_scale<4>(bta_ref, 1.5).perform();

    ctf_btod_distribute<4>(bta).perform(dbta);
    ctf_btod_scale<4>(dbta, 1.5).perform();
    ctf_btod_collect<4>(dbta).perform(bta);

    compare_ref<4>::compare(testname, bta, bta_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

