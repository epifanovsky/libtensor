#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/block_tensor/btod_symmetrize2.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_copy.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include <libtensor/ctf_block_tensor/ctf_btod_symmetrize2.h>
#include "../compare_ref.h"
#include "ctf_btod_symmetrize2_test.h"

namespace libtensor {


void ctf_btod_symmetrize2_test::perform() throw(libtest::test_exception) {

    ctf::init();

    try {

        test_1();
        test_2();
        test_3();
        test_4();
        test_5();

    } catch(...) {
        ctf::exit();
        throw;
    }

    ctf::exit();
}


void ctf_btod_symmetrize2_test::test_1() {

    static const char testname[] = "ctf_btod_symmetrize2_test::test_1()";

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

    btod_copy<2> cp(bta);
    btod_symmetrize2<2>(cp, 0, 1, true).perform(btb_ref);

    ctf_btod_distribute<2>(bta).perform(dbta);
    ctf_btod_distribute<2>(btb).perform(dbtb);

    ctf_btod_copy<2> dcp(dbta);
    ctf_btod_symmetrize2<2>(dcp, 0, 1, true).perform(dbtb);
    ctf_btod_collect<2>(dbtb).perform(btb);

    compare_ref<2>::compare(testname, btb, btb_ref, 1e-15);

    // check block symmetry

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_symmetrize2_test::test_2() {

    static const char testname[] = "ctf_btod_symmetrize2_test::test_2()";

    typedef std_allocator<double> allocator_t;

    try {

    mask<4> m1111;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

    index<4> i1, i2;
    i2[0] = 19; i2[1] = 19; i2[2] = 19; i2[3] = 19;
    dimensions<4> dimsa(index_range<4>(i1, i2)), dimsb(dimsa);
    block_index_space<4> bisa(dimsa);
    bisa.split(m1111, 6);
    block_index_space<4> bisb(bisa);

    block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btb_ref(bisb);
    ctf_block_tensor<4, double> dbta(bisa), dbtb(bisb);

    btod_random<4>().perform(bta);
    btod_random<4>().perform(btb);

    btod_copy<4> cp(bta);
    btod_symmetrize2<4>(cp, 0, 1, false).perform(btb_ref);

    ctf_btod_distribute<4>(bta).perform(dbta);
    ctf_btod_distribute<4>(btb).perform(dbtb);

    ctf_btod_copy<4> dcp(dbta);
    ctf_btod_symmetrize2<4>(dcp, 0, 1, false).perform(dbtb);
    ctf_btod_collect<4>(dbtb).perform(btb);

    compare_ref<4>::compare(testname, btb, btb_ref, 1e-15);

    // check block symmetry

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_symmetrize2_test::test_3() {

    static const char testname[] = "ctf_btod_symmetrize2_test::test_3()";

    typedef std_allocator<double> allocator_t;

    try {

    mask<4> m1111;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

    index<4> i1, i2;
    i2[0] = 19; i2[1] = 19; i2[2] = 19; i2[3] = 19;
    dimensions<4> dimsa(index_range<4>(i1, i2)), dimsb(dimsa);
    block_index_space<4> bisa(dimsa);
    bisa.split(m1111, 6);
    block_index_space<4> bisb(bisa);

    block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btb_ref(bisb);
    ctf_block_tensor<4, double> dbta(bisa), dbtb(bisb);

    btod_random<4>().perform(bta);
    btod_random<4>().perform(btb);

    btod_copy<4> cp(bta);
    btod_symmetrize2<4> sym1(cp, 0, 1, false);
    btod_symmetrize2<4>(sym1, 2, 3, true).perform(btb_ref);

    ctf_btod_distribute<4>(bta).perform(dbta);
    ctf_btod_distribute<4>(btb).perform(dbtb);

    ctf_btod_copy<4> dcp(dbta);
    ctf_btod_symmetrize2<4> dsym1(dcp, 0, 1, false);
    ctf_btod_symmetrize2<4>(dsym1, 2, 3, true).perform(dbtb);
    ctf_btod_collect<4>(dbtb).perform(btb);

    compare_ref<4>::compare(testname, btb, btb_ref, 1e-15);

    // check block symmetry

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_symmetrize2_test::test_4() {

    static const char testname[] = "ctf_btod_symmetrize2_test::test_4()";

    typedef std_allocator<double> allocator_t;

    try {

    mask<4> m1111;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

    index<4> i1, i2;
    i2[0] = 19; i2[1] = 19; i2[2] = 19; i2[3] = 19;
    dimensions<4> dimsa(index_range<4>(i1, i2)), dimsb(dimsa);
    block_index_space<4> bisa(dimsa);
    bisa.split(m1111, 6);
    block_index_space<4> bisb(bisa);

    block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btb_ref(bisb);
    ctf_block_tensor<4, double> dbta(bisa), dbtb(bisb);

    btod_random<4>().perform(bta);
    btod_random<4>().perform(btb);

    permutation<4> perm;
    perm.permute(0, 1).permute(2, 3);

    btod_copy<4> cp(bta);
    btod_symmetrize2<4>(cp, perm, true).perform(btb_ref);

    ctf_btod_distribute<4>(bta).perform(dbta);
    ctf_btod_distribute<4>(btb).perform(dbtb);

    ctf_btod_copy<4> dcp(dbta);
    ctf_btod_symmetrize2<4>(dcp, perm, true).perform(dbtb);
    ctf_btod_collect<4>(dbtb).perform(btb);

    compare_ref<4>::compare(testname, btb, btb_ref, 1e-15);

    // check block symmetry

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_symmetrize2_test::test_5() {

    static const char testname[] = "ctf_btod_symmetrize2_test::test_5()";

    typedef std_allocator<double> allocator_t;

    try {

    mask<4> m1111;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dimsa(index_range<4>(i1, i2)), dimsb(dimsa);
    block_index_space<4> bisa(dimsa);
    bisa.split(m1111, 6);
    block_index_space<4> bisb(bisa);

    block_tensor<4, double, allocator_t> bta(bisa), btb(bisb), btb_ref(bisb);
    ctf_block_tensor<4, double> dbta(bisa), dbtb(bisb);

    btod_random<4>().perform(bta);
    btod_random<4>().perform(btb);

    permutation<4> perm;
    perm.permute(0, 2).permute(1, 3);

    btod_copy<4> cp(bta);
    btod_symmetrize2<4>(cp, perm, true).perform(btb_ref);

    ctf_btod_distribute<4>(bta).perform(dbta);
    ctf_btod_distribute<4>(btb).perform(dbtb);

    ctf_btod_copy<4> dcp(dbta);
    ctf_btod_symmetrize2<4>(dcp, perm, true).perform(dbtb);
    ctf_btod_collect<4>(dbtb).perform(btb);

    compare_ref<4>::compare(testname, btb, btb_ref, 1e-15);

    // check block symmetry

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

