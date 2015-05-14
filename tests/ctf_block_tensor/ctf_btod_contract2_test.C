#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/block_tensor/btod_contract2.h>
#include <libtensor/block_tensor/btod_copy.h>
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
        test_2(0.0);
        test_2(1.5);
        test_3(0.0);
        test_3(-0.2);
        test_4(0.0);
        test_4(1.45);

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


void ctf_btod_contract2_test::test_2(double d) {

    std::ostringstream tnss;
    tnss << "ctf_btod_contract2_test::test_2(" << d << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

    mask<2> m11;
    m11[0] = true; m11[1] = true;
    mask<4> m1111;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

    index<2> i2a, i2b;
    i2b[0] = 19; i2b[1] = 19;
    index<4> i4a, i4b;
    i4b[0] = 19; i4b[1] = 19; i4b[2] = 19; i4b[3] = 19;
    dimensions<2> dimsa(index_range<2>(i2a, i2b));
    dimensions<4> dimsc(index_range<4>(i4a, i4b));
    block_index_space<2> bisa(dimsa);
    bisa.split(m11, 6);
    block_index_space<2> bisb(bisa);
    block_index_space<4> bisc(dimsc);
    bisc.split(m1111, 6);

    block_tensor<2, double, allocator_t> bta(bisa), btb(bisb);
    block_tensor<4, double, allocator_t> btc(bisc), btc_ref(bisc);
    ctf_block_tensor<2, double> dbta(bisa), dbtb(bisb);
    ctf_block_tensor<4, double> dbtc(bisc);

    {
        se_perm<2, double> seperm2(permutation<2>().permute(0, 1),
            scalar_transf<double>());
        se_perm<4, double> seperm4(permutation<4>().permute(2, 3),
            scalar_transf<double>());
        block_tensor_ctrl<2, double> ca(bta), cb(btb);
        ca.req_symmetry().insert(seperm2);
        cb.req_symmetry().insert(seperm2);
        block_tensor_ctrl<4, double> cc(btc);
        cc.req_symmetry().insert(seperm4);
    }

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);
    btod_random<4>().perform(btc);
    btod_copy<4>(btc).perform(btc_ref);

    ctf_btod_distribute<2>(bta).perform(dbta);
    ctf_btod_distribute<2>(btb).perform(dbtb);
    ctf_btod_distribute<4>(btc).perform(dbtc);

    contraction2<2, 2, 0> contr;

    if(d == 0.0) {
        btod_contract2<2, 2, 0>(contr, bta, btb).perform(btc_ref);
        ctf_btod_contract2<2, 2, 0>(contr, dbta, dbtb).perform(dbtc);
    } else {
        btod_contract2<2, 2, 0>(contr, bta, btb).perform(btc_ref, d);
        ctf_btod_contract2<2, 2, 0>(contr, dbta, dbtb).perform(dbtc, d);
    }

    ctf_btod_collect<4>(dbtc).perform(btc);

    compare_ref<4>::compare(tn.c_str(), btc, btc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_contract2_test::test_3(double d) {

    std::ostringstream tnss;
    tnss << "ctf_btod_contract2_test::test_3(" << d << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

    mask<2> m11;
    m11[0] = true; m11[1] = true;
    mask<4> m1111;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

    index<2> i2a, i2b;
    i2b[0] = 19; i2b[1] = 19;
    index<4> i4a, i4b;
    i4b[0] = 19; i4b[1] = 19; i4b[2] = 19; i4b[3] = 19;
    dimensions<2> dimsa(index_range<2>(i2a, i2b));
    dimensions<4> dimsc(index_range<4>(i4a, i4b));
    block_index_space<2> bisa(dimsa);
    bisa.split(m11, 6);
    block_index_space<4> bisc(dimsc);
    bisc.split(m1111, 6);

    block_tensor<2, double, allocator_t> bta(bisa);
    block_tensor<4, double, allocator_t> btc(bisc), btc_ref(bisc);
    ctf_block_tensor<2, double> dbta(bisa);
    ctf_block_tensor<4, double> dbtc(bisc);

    {
        se_perm<2, double> seperm2(permutation<2>().permute(0, 1),
            scalar_transf<double>());
        se_perm<4, double> seperm4(permutation<4>().permute(2, 3),
            scalar_transf<double>());
        block_tensor_ctrl<2, double> ca(bta);
        ca.req_symmetry().insert(seperm2);
        block_tensor_ctrl<4, double> cc(btc);
        cc.req_symmetry().insert(seperm4);
    }

    btod_random<2>().perform(bta);
    btod_random<4>().perform(btc);
    btod_copy<4>(btc).perform(btc_ref);

    ctf_btod_distribute<2>(bta).perform(dbta);
    ctf_btod_distribute<4>(btc).perform(dbtc);

    contraction2<2, 2, 0> contr;

    if(d == 0) {
        btod_contract2<2, 2, 0>(contr, bta, bta).perform(btc_ref);
        ctf_btod_contract2<2, 2, 0>(contr, dbta, dbta).perform(dbtc);
    } else {
        btod_contract2<2, 2, 0>(contr, bta, bta).perform(btc_ref, d);
        ctf_btod_contract2<2, 2, 0>(contr, dbta, dbta).perform(dbtc, d);
    }

    ctf_btod_collect<4>(dbtc).perform(btc);

    compare_ref<4>::compare(tn.c_str(), btc, btc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void ctf_btod_contract2_test::test_4(double d) {

    std::ostringstream tnss;
    tnss << "ctf_btod_contract2_test::test_4(" << d << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

    mask<2> m11;
    m11[0] = true; m11[1] = true;
    mask<4> m1111;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

    index<2> i2a, i2b;
    i2b[0] = 19; i2b[1] = 19;
    index<4> i4a, i4b;
    i4b[0] = 19; i4b[1] = 19; i4b[2] = 19; i4b[3] = 19;
    dimensions<2> dimsa(index_range<2>(i2a, i2b));
    dimensions<4> dimsb(index_range<4>(i4a, i4b));
    dimensions<2> dimsc(index_range<2>(i2a, i2b));
    block_index_space<2> bisa(dimsa);
    bisa.split(m11, 6);
    block_index_space<4> bisb(dimsb);
    bisb.split(m1111, 6);
    block_index_space<2> bisc(bisa);

    block_tensor<2, double, allocator_t> bta(bisa);
    block_tensor<4, double, allocator_t> btb(bisb);
    block_tensor<2, double, allocator_t> btc(bisc), btc_ref(bisc);
    ctf_block_tensor<2, double> dbta(bisa);
    ctf_block_tensor<4, double> dbtb(bisb);
    ctf_block_tensor<2, double> dbtc(bisc);

    {
        se_perm<2, double> seperm2(permutation<2>().permute(0, 1),
            scalar_transf<double>());
        se_perm<4, double> seperm4(permutation<4>().permute(0, 1).permute(2, 3),
            scalar_transf<double>());
        block_tensor_ctrl<2, double> ca(bta);
        ca.req_symmetry().insert(seperm2);
        block_tensor_ctrl<4, double> cb(btb);
        cb.req_symmetry().insert(seperm4);
        block_tensor_ctrl<2, double> cc(btc);
        cc.req_symmetry().insert(seperm2);
    }

    btod_random<2>().perform(bta);
    btod_random<4>().perform(btb);
    btod_random<2>().perform(btc);
    btod_copy<2>(btc).perform(btc_ref);

    ctf_btod_distribute<2>(bta).perform(dbta);
    ctf_btod_distribute<4>(btb).perform(dbtb);
    ctf_btod_distribute<2>(btc).perform(dbtc);

    contraction2<0, 2, 2> contr;
    contr.contract(0, 2);
    contr.contract(1, 3);

    if(d == 0) {
        btod_contract2<0, 2, 2>(contr, bta, btb).perform(btc_ref);
        ctf_btod_contract2<0, 2, 2>(contr, dbta, dbtb).perform(dbtc);
    } else {
        btod_contract2<0, 2, 2>(contr, bta, btb).perform(btc_ref, d);
        ctf_btod_contract2<0, 2, 2>(contr, dbta, dbtb).perform(dbtc, d);
    }

    ctf_btod_collect<2>(dbtc).perform(btc);

    compare_ref<2>::compare(tn.c_str(), btc, btc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

