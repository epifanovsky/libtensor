#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_copy.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include "../compare_ref.h"
#include "ctf_btod_copy_test.h"

namespace libtensor {


void ctf_btod_copy_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1a();
        test_1b();
        test_2a();
        test_2b();
        test_3a();
        test_3b();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_btod_copy_test::test_1a() {

    static const char testname[] = "ctf_btod_copy_test::test_1a()";

    typedef std_allocator<double> allocator_t;

    try {

    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    block_index_space<2> bisa(dimsa);
    bisa.split(m11, 40);
    block_index_space<2> bisb(bisa);

    block_tensor<2, double, allocator_t> bta(bisa), btb(bisb), btb_ref(bisb);
    ctf_block_tensor<2, double> dbta(bisa), dbtb(bisb);

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);
    btod_copy<2>(bta).perform(btb_ref);

    ctf_btod_distribute<2>(bta).perform(dbta);
    ctf_btod_distribute<2>(btb).perform(dbtb);

    ctf_btod_copy<2>(dbta).perform(dbtb);
    ctf_btod_collect<2>(dbtb).perform(btb);

    compare_ref<2>::compare(testname, btb, btb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_copy_test::test_1b() {

    static const char testname[] = "ctf_btod_copy_test::test_1b()";

    typedef std_allocator<double> allocator_t;

    try {

    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    block_index_space<2> bisa(dimsa);
    bisa.split(m11, 40);
    block_index_space<2> bisb(bisa);

    block_tensor<2, double, allocator_t> bta(bisa), btb(bisb), btb_ref(bisb);
    ctf_block_tensor<2, double> dbta(bisa), dbtb(bisb);

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);
    btod_copy<2>(btb).perform(btb_ref);
    btod_copy<2>(bta, -0.5).perform(btb_ref, 1.0);

    ctf_btod_distribute<2>(bta).perform(dbta);
    ctf_btod_distribute<2>(btb).perform(dbtb);

    ctf_btod_copy<2>(dbta, -0.5).perform(dbtb, 1.0);
    ctf_btod_collect<2>(dbtb).perform(btb);

    compare_ref<2>::compare(testname, btb, btb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_copy_test::test_2a() {

    static const char testname[] = "ctf_btod_copy_test::test_2a()";

    typedef std_allocator<double> allocator_t;

    try {

    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    block_index_space<2> bisa(dimsa);
    bisa.split(m11, 40);
    block_index_space<2> bisb(bisa);
    permutation<2> perma;
    perma.permute(0, 1);
    bisb.permute(perma);

    block_tensor<2, double, allocator_t> bta(bisa), btb(bisb), btb_ref(bisb);
    ctf_block_tensor<2, double> dbta(bisa), dbtb(bisb);

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);
    btod_copy<2>(bta, perma).perform(btb_ref);

    ctf_btod_distribute<2>(bta).perform(dbta);
    ctf_btod_distribute<2>(btb).perform(dbtb);

    ctf_btod_copy<2>(dbta, perma).perform(dbtb);
    ctf_btod_collect<2>(dbtb).perform(btb);

    compare_ref<2>::compare(testname, btb, btb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_copy_test::test_2b() {

    static const char testname[] = "ctf_btod_copy_test::test_2b()";

    typedef std_allocator<double> allocator_t;

    try {

    mask<2> m11;
    m11[0] = true; m11[1] = true;

    index<2> i1, i2;
    i2[0] = 99; i2[1] = 99;
    dimensions<2> dimsa(index_range<2>(i1, i2)), dimsb(dimsa);
    block_index_space<2> bisa(dimsa);
    bisa.split(m11, 40);
    block_index_space<2> bisb(bisa);
    permutation<2> perma;
    perma.permute(0, 1);
    bisb.permute(perma);

    block_tensor<2, double, allocator_t> bta(bisa), btb(bisb), btb_ref(bisb);
    ctf_block_tensor<2, double> dbta(bisa), dbtb(bisb);

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);
    btod_copy<2>(btb).perform(btb_ref);
    btod_copy<2>(bta, perma, -0.5).perform(btb_ref, 1.0);

    ctf_btod_distribute<2>(bta).perform(dbta);
    ctf_btod_distribute<2>(btb).perform(dbtb);

    ctf_btod_copy<2>(dbta, perma).perform(dbtb, -0.5);
    ctf_btod_collect<2>(dbtb).perform(btb);

    compare_ref<2>::compare(testname, btb, btb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_copy_test::test_3a() {

    static const char testname[] = "ctf_btod_copy_test::test_3a()";

    typedef std_allocator<double> allocator_t;

    try {

    mask<3> m100, m010;
    m100[0] = true; m010[1] = true;

    index<3> i1, i2;
    i2[0] = 19; i2[1] = 9; i2[2] = 7;
    dimensions<3> dimsa(index_range<3>(i1, i2)), dimsb(dimsa);
    block_index_space<3> bisa(dimsa);
    bisa.split(m100, 9);
    bisa.split(m010, 3);
    block_index_space<3> bisb(bisa);
    permutation<3> perma;
    perma.permute(0, 1).permute(1, 2);
    bisb.permute(perma);

    block_tensor<3, double, allocator_t> bta(bisa), btb(bisb), btb_ref(bisb);
    ctf_block_tensor<3, double> dbta(bisa), dbtb(bisb);

    btod_random<3>().perform(bta);
    btod_random<3>().perform(btb);
    btod_copy<3>(bta, perma, 0.5).perform(btb_ref);

    ctf_btod_distribute<3>(bta).perform(dbta);
    ctf_btod_distribute<3>(btb).perform(dbtb);

    ctf_btod_copy<3>(dbta, perma, 0.5).perform(dbtb);
    ctf_btod_collect<3>(dbtb).perform(btb);

    compare_ref<3>::compare(testname, btb, btb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_copy_test::test_3b() {

    static const char testname[] = "ctf_btod_copy_test::test_3b()";

    typedef std_allocator<double> allocator_t;

    try {

    mask<3> m100, m010;
    m100[0] = true; m010[1] = true;

    index<3> i1, i2;
    i2[0] = 19; i2[1] = 9; i2[2] = 7;
    dimensions<3> dimsa(index_range<3>(i1, i2)), dimsb(dimsa);
    block_index_space<3> bisa(dimsa);
    bisa.split(m100, 9);
    bisa.split(m010, 3);
    block_index_space<3> bisb(bisa);
    permutation<3> perma;
    perma.permute(0, 2).permute(1, 2);
    bisb.permute(perma);

    block_tensor<3, double, allocator_t> bta(bisa), btb(bisb), btb_ref(bisb);
    ctf_block_tensor<3, double> dbta(bisa), dbtb(bisb);

    btod_random<3>().perform(bta);
    btod_random<3>().perform(btb);
    btod_copy<3>(btb).perform(btb_ref);
    btod_copy<3>(bta, perma, -1.0).perform(btb_ref, 1.0);

    ctf_btod_distribute<3>(bta).perform(dbta);
    ctf_btod_distribute<3>(btb).perform(dbtb);

    ctf_btod_copy<3>(dbta, perma, -1.0).perform(dbtb, 1.0);
    ctf_btod_collect<3>(dbtb).perform(btb);

    compare_ref<3>::compare(testname, btb, btb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

