#include <algorithm>
#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_unfold_symmetry.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include "../compare_ref.h"
#include "gen_bto_unfold_symmetry_test.h"

namespace libtensor {


void gen_bto_unfold_symmetry_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 65536, 65536);

    try {

    test_1();
    test_2();
    test_3();
    test_4();

    } catch (...) {
        allocator<double>::shutdown();
        throw;
    }
    allocator<double>::shutdown();
}


void gen_bto_unfold_symmetry_test::test_1() {

    static const char *testname = "gen_bto_unfold_symmetry_test::test_1()";

    typedef std_allocator<double> allocator_type;

    try {

    index<2> i1, i2;
    i2[0] = 10; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    dimensions<2> bidims(bis.get_block_index_dims());

    symmetry<2, double> syma(bis);
    se_perm<2, double> se1(permutation<2>().permute(0, 1),
        scalar_transf<double>(1.0));
    syma.insert(se1);

    block_tensor<2, double, allocator_type> bta(bis), btb(bis);

    {
        block_tensor_wr_ctrl<2, double> ca(bta);
        so_copy<2, double>(syma).perform(ca.req_symmetry());
    }

    btod_random<2>().perform(bta);
    bta.set_immutable();
    btod_copy<2>(bta).perform(btb);

    gen_bto_unfold_symmetry<2, btod_traits>().perform(btb);

    {
        block_tensor_rd_ctrl<2, double> cb(btb);
        const symmetry<2, double> &symb = cb.req_const_symmetry();
        if(symb.begin() != symb.end()) {
            fail_test(testname, __FILE__, __LINE__,
                "Symmetry of B is not empty");
        }
    }

    dense_tensor<2, double, allocator_type> tb(dims), tb_ref(dims);
    tod_btconv<2>(bta).perform(tb_ref);
    tod_btconv<2>(btb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void gen_bto_unfold_symmetry_test::test_2() {

    static const char *testname = "gen_bto_unfold_symmetry_test::test_2()";

    typedef std_allocator<double> allocator_type;

    try {

    index<2> i1, i2;
    i2[0] = 10; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    bis.split(m11, 2);
    bis.split(m11, 5);
    dimensions<2> bidims(bis.get_block_index_dims());

    symmetry<2, double> syma(bis);
    se_perm<2, double> se1(permutation<2>().permute(0, 1),
        scalar_transf<double>(-1.0));
    syma.insert(se1);

    block_tensor<2, double, allocator_type> bta(bis), btb(bis);

    {
        block_tensor_wr_ctrl<2, double> ca(bta);
        so_copy<2, double>(syma).perform(ca.req_symmetry());
    }

    btod_random<2>().perform(bta);
    bta.set_immutable();
    btod_copy<2>(bta).perform(btb);

    gen_bto_unfold_symmetry<2, btod_traits>().perform(btb);

    {
        block_tensor_rd_ctrl<2, double> cb(btb);
        const symmetry<2, double> &symb = cb.req_const_symmetry();
        if(symb.begin() != symb.end()) {
            fail_test(testname, __FILE__, __LINE__,
                "Symmetry of B is not empty");
        }
    }

    dense_tensor<2, double, allocator_type> tb(dims), tb_ref(dims);
    tod_btconv<2>(bta).perform(tb_ref);
    tod_btconv<2>(btb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void gen_bto_unfold_symmetry_test::test_3() {

    static const char *testname = "gen_bto_unfold_symmetry_test::test_3()";

    typedef std_allocator<double> allocator_type;

    try {

    index<2> i1, i2;
    i2[0] = 10; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    bis.split(m11, 2);
    bis.split(m11, 5);
    dimensions<2> bidims(bis.get_block_index_dims());

    symmetry<2, double> syma(bis);
    se_perm<2, double> se1(permutation<2>().permute(0, 1),
        scalar_transf<double>(-1.0));
    syma.insert(se1);

    block_tensor<2, double, allocator_type> bta(bis), bta1(bis), btb(bis);

    {
        block_tensor_wr_ctrl<2, double> ca(bta);
        so_copy<2, double>(syma).perform(ca.req_symmetry());
    }

    btod_random<2>().perform(bta);
    bta.set_immutable();
    btod_copy<2>(bta).perform(btb);
    {
        block_tensor_wr_ctrl<2, double> cb(btb);
        cb.req_symmetry().clear();
    }

    std::vector<size_t> blst;
    blst.push_back(1);
    blst.push_back(2);
    blst.push_back(6);

    gen_bto_unfold_symmetry<2, btod_traits>().perform(syma, blst, btb);

    {
        block_tensor_rd_ctrl<2, double> cb(btb);
        const symmetry<2, double> &symb = cb.req_const_symmetry();
        if(symb.begin() != symb.end()) {
            fail_test(testname, __FILE__, __LINE__,
                "Symmetry of B is not empty");
        }
    }

    btod_copy<2>(bta).perform(bta1, 1.0);

    {
        block_tensor_wr_ctrl<2, double> ca1(bta1);
        abs_index<2> ai(bidims);
        do {
            if(std::find(blst.begin(), blst.end(), ai.get_abs_index()) !=
                blst.end()) continue;
            orbit<2, double> o(syma, ai.get_index());
            if(ai.get_abs_index() == o.get_acindex()) continue;
            ca1.req_zero_block(ai.get_index());
        } while(ai.inc());
    }

    dense_tensor<2, double, allocator_type> tb(dims), tb_ref(dims);
    tod_btconv<2>(bta1).perform(tb_ref);
    tod_btconv<2>(btb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void gen_bto_unfold_symmetry_test::test_4() {

    static const char *testname = "gen_bto_unfold_symmetry_test::test_4()";

    typedef std_allocator<double> allocator_type;

    try {

    index<2> i1, i2;
    i2[0] = 10; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    bis.split(m11, 2);
    bis.split(m11, 5);
    dimensions<2> bidims(bis.get_block_index_dims());

    symmetry<2, double> syma(bis);
    se_perm<2, double> se1(permutation<2>().permute(0, 1),
        scalar_transf<double>(1.0));
    syma.insert(se1);

    block_tensor<2, double, allocator_type> bta(bis), bta1(bis), btb(bis);

    {
        block_tensor_wr_ctrl<2, double> ca(bta);
        so_copy<2, double>(syma).perform(ca.req_symmetry());
    }

    btod_random<2>().perform(bta);
    {
        index<2> i01, i02, i22;
        i01[0] = 0; i01[1] = 1;
        i02[0] = 0; i02[1] = 2;
        i22[0] = 0; i22[1] = 2;
        block_tensor_wr_ctrl<2, double> ca(bta);
        ca.req_zero_block(i01);
        ca.req_zero_block(i02);
        ca.req_zero_block(i22);
    }
    bta.set_immutable();
    btod_copy<2>(bta).perform(btb);
    {
        block_tensor_wr_ctrl<2, double> cb(btb);
        cb.req_symmetry().clear();
    }

    std::vector<size_t> blst;
    blst.push_back(1);
    blst.push_back(2);
    blst.push_back(6);
    blst.push_back(4);
    blst.push_back(5);

    gen_bto_unfold_symmetry<2, btod_traits>().perform(syma, blst, btb);

    {
        block_tensor_rd_ctrl<2, double> cb(btb);
        const symmetry<2, double> &symb = cb.req_const_symmetry();
        if(symb.begin() != symb.end()) {
            fail_test(testname, __FILE__, __LINE__,
                "Symmetry of B is not empty");
        }
    }

    btod_copy<2>(bta).perform(bta1, 1.0);

    {
        block_tensor_wr_ctrl<2, double> ca1(bta1);
        abs_index<2> ai(bidims);
        do {
            if(std::find(blst.begin(), blst.end(), ai.get_abs_index()) !=
                blst.end()) continue;
            orbit<2, double> o(syma, ai.get_index());
            if(ai.get_abs_index() == o.get_acindex()) continue;
            ca1.req_zero_block(ai.get_index());
        } while(ai.inc());
    }

    dense_tensor<2, double, allocator_type> tb(dims), tb_ref(dims);
    tod_btconv<2>(bta1).perform(tb_ref);
    tod_btconv<2>(btb).perform(tb);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


} // namespace libtensor
