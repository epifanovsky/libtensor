#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include "../compare_ref.h"
#include "gen_bto_aux_add_test.h"

namespace libtensor {


void gen_bto_aux_add_test::perform() throw(libtest::test_exception) {

    allocator<double>::vmm().init(16, 16, 65536, 65536);

    try {

    test_1a();
    test_1b();
    test_1c();
    test_2();
    test_3a();
    test_3b();

    } catch (...) {
        allocator<double>::vmm().shutdown();
        throw;
    }
    allocator<double>::vmm().shutdown();
}


void gen_bto_aux_add_test::test_1a() {

    static const char *testname = "gen_bto_aux_add_test::test_1a()";

    typedef std_allocator<double> allocator_type;
    typedef block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i1, i2;
    i2[0] = 11; i2[1] = 11;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    bis.split(m11, 2);
    bis.split(m11, 5);
    dimensions<2> bidims(bis.get_block_index_dims());

    symmetry<2, double> syma(bis), symb(bis);

    index<2> i00, i12;
    i12[0] = 1; i12[1] = 2;

    block_tensor<2, double, allocator_type> bta(bis), btb(bis), btc(bis),
        btc_ref(bis);

    //  Set up initial A

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> ca(bta);
        so_copy<2, double>(syma).perform(ca.req_symmetry());
    }
    bta.set_immutable();

    //  Set up initial B

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> cb(btb);
        dense_tensor_wr_i<2, double> &b00 = cb.req_block(i00);
        tod_random<2>().perform(b00);
        cb.ret_block(i00);
        dense_tensor_wr_i<2, double> &b12 = cb.req_block(i12);
        tod_random<2>().perform(b12);
        cb.ret_block(i12);
    }
    btb.set_immutable();

    //  Build schedules

    assignment_schedule<2, double> sch(bidims);
    sch.insert(i00);
    sch.insert(i12);

    addition_schedule<2, btod_traits> asch(syma, symb);
    {
        gen_block_tensor_rd_ctrl<2, bti_traits> ca(bta);
        asch.build(sch, ca);
    }

    //  Send blocks to the addition stream

    gen_bto_aux_add<2, btod_traits> out(syma, asch, btc,
        scalar_transf<double>(1.0));
    {
        gen_block_tensor_rd_ctrl<2, bti_traits> cb(btb);
        gen_block_tensor_wr_ctrl<2, bti_traits> cc(btc_ref);
        tensor_transf<2, double> tr0;
        out.open();

        dense_tensor_rd_i<2, double> &b00 = cb.req_const_block(i00);
        dense_tensor_wr_i<2, double> &c00 = cc.req_block(i00);
        tod_copy<2>(b00).perform(true, c00);
        out.put(i00, b00, tr0);
        cc.ret_block(i00);
        cb.ret_const_block(i00);

        dense_tensor_rd_i<2, double> &b12 = cb.req_const_block(i12);
        dense_tensor_wr_i<2, double> &c12 = cc.req_block(i12);
        tod_copy<2>(b12).perform(true, c12);
        out.put(i12, b12, tr0);
        cc.ret_block(i12);
        cb.ret_const_block(i12);

        out.close();
    }

    //  Compare against reference

    dense_tensor<2, double, allocator_type> tc(dims), tc_ref(dims);
    tod_btconv<2>(btc).perform(tc);
    tod_btconv<2>(btc_ref).perform(tc_ref);

    compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void gen_bto_aux_add_test::test_1b() {

    static const char *testname = "gen_bto_aux_add_test::test_1b()";

    typedef std_allocator<double> allocator_type;
    typedef block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i1, i2;
    i2[0] = 11; i2[1] = 11;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    bis.split(m11, 2);
    bis.split(m11, 5);
    dimensions<2> bidims(bis.get_block_index_dims());

    symmetry<2, double> syma(bis), symb(bis);

    index<2> i00, i12;
    i12[0] = 1; i12[1] = 2;

    block_tensor<2, double, allocator_type> bta(bis), btb1(bis), btb2(bis),
        btc(bis), btc_ref(bis);

    //  Set up initial A

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> ca(bta);
        so_copy<2, double>(syma).perform(ca.req_symmetry());
    }
    bta.set_immutable();

    //  Set up initial B

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> cb(btb1);
        dense_tensor_wr_i<2, double> &b00 = cb.req_block(i00);
        tod_random<2>().perform(b00);
        cb.ret_block(i00);
        dense_tensor_wr_i<2, double> &b12 = cb.req_block(i12);
        tod_random<2>().perform(b12);
        cb.ret_block(i12);
    }
    {
        gen_block_tensor_wr_ctrl<2, bti_traits> cb(btb2);
        dense_tensor_wr_i<2, double> &b12 = cb.req_block(i12);
        tod_random<2>().perform(b12);
        cb.ret_block(i12);
    }
    btb1.set_immutable();
    btb2.set_immutable();

    //  Build schedules

    assignment_schedule<2, double> sch(bidims);
    sch.insert(i00);
    sch.insert(i12);

    addition_schedule<2, btod_traits> asch(syma, symb);
    {
        gen_block_tensor_rd_ctrl<2, bti_traits> ca(bta);
        asch.build(sch, ca);
    }

    //  Send blocks to the addition stream

    gen_bto_aux_add<2, btod_traits> out(syma, asch, btc,
        scalar_transf<double>(1.0));
    {
        gen_block_tensor_rd_ctrl<2, bti_traits> cb1(btb1);
        gen_block_tensor_rd_ctrl<2, bti_traits> cb2(btb2);
        gen_block_tensor_wr_ctrl<2, bti_traits> cc(btc_ref);
        tensor_transf<2, double> tr0;
        out.open();

        {
            dense_tensor_rd_i<2, double> &b00 = cb1.req_const_block(i00);
            dense_tensor_wr_i<2, double> &c00 = cc.req_block(i00);
            tod_copy<2>(b00).perform(true, c00);
            out.put(i00, b00, tr0);
            cc.ret_block(i00);
            cb1.ret_const_block(i00);
        }

        {
            dense_tensor_rd_i<2, double> &b12 = cb1.req_const_block(i12);
            dense_tensor_wr_i<2, double> &c12 = cc.req_block(i12);
            tod_copy<2>(b12).perform(true, c12);
            out.put(i12, b12, tr0);
            cc.ret_block(i12);
            cb1.ret_const_block(i12);
        }

        {
            dense_tensor_rd_i<2, double> &b12 = cb2.req_const_block(i12);
            dense_tensor_wr_i<2, double> &c12 = cc.req_block(i12);
            tod_copy<2>(b12).perform(false, c12);
            out.put(i12, b12, tr0);
            cc.ret_block(i12);
            cb2.ret_const_block(i12);
        }
        out.close();
    }

    //  Compare against reference

    dense_tensor<2, double, allocator_type> tc(dims), tc_ref(dims);
    tod_btconv<2>(btc).perform(tc);
    tod_btconv<2>(btc_ref).perform(tc_ref);

    compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void gen_bto_aux_add_test::test_1c() {

    static const char *testname = "gen_bto_aux_add_test::test_1c()";

    typedef std_allocator<double> allocator_type;
    typedef block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i1, i2;
    i2[0] = 11; i2[1] = 11;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    bis.split(m11, 2);
    bis.split(m11, 5);
    dimensions<2> bidims(bis.get_block_index_dims());

    symmetry<2, double> syma(bis), symb(bis);

    index<2> i00, i12;
    i12[0] = 1; i12[1] = 2;

    block_tensor<2, double, allocator_type> bta(bis), btb1(bis), btb2(bis),
        btc(bis), btc_ref(bis);

    //  Set up initial A

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> ca(bta);
        so_copy<2, double>(syma).perform(ca.req_symmetry());
    }
    bta.set_immutable();

    //  Set up initial B

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> cb(btb1);
        dense_tensor_wr_i<2, double> &b00 = cb.req_block(i00);
        tod_random<2>().perform(b00);
        cb.ret_block(i00);
        dense_tensor_wr_i<2, double> &b12 = cb.req_block(i12);
        tod_random<2>().perform(b12);
        cb.ret_block(i12);
    }
    {
        gen_block_tensor_wr_ctrl<2, bti_traits> cb(btb2);
        dense_tensor_wr_i<2, double> &b12 = cb.req_block(i12);
        tod_random<2>().perform(b12);
        cb.ret_block(i12);
    }
    btb1.set_immutable();
    btb2.set_immutable();

    //  Build schedules

    assignment_schedule<2, double> sch(bidims);
    sch.insert(i00);
    sch.insert(i12);

    addition_schedule<2, btod_traits> asch(syma, symb);
    {
        gen_block_tensor_rd_ctrl<2, bti_traits> ca(bta);
        asch.build(sch, ca);
    }

    //  Send blocks to the addition stream

    gen_bto_aux_add<2, btod_traits> out(syma, asch, btc,
        scalar_transf<double>(1.0));
    {
        gen_block_tensor_rd_ctrl<2, bti_traits> cb1(btb1);
        gen_block_tensor_rd_ctrl<2, bti_traits> cb2(btb2);
        gen_block_tensor_wr_ctrl<2, bti_traits> cc(btc_ref);
        tensor_transf<2, double> tr0;

        out.open();

        {
            dense_tensor_rd_i<2, double> &b00 = cb1.req_const_block(i00);
            dense_tensor_wr_i<2, double> &c00 = cc.req_block(i00);
            tod_copy<2>(b00).perform(true, c00);
            out.put(i00, b00, tr0);
            cc.ret_block(i00);
            cb1.ret_const_block(i00);
        }

        {
            dense_tensor_rd_i<2, double> &b12 = cb1.req_const_block(i12);
            dense_tensor_wr_i<2, double> &c12 = cc.req_block(i12);
            tod_copy<2>(b12).perform(true, c12);
            out.put(i12, b12, tr0);
            cc.ret_block(i12);
            cb1.ret_const_block(i12);
        }

        out.close();
        out.open();

        {
            dense_tensor_rd_i<2, double> &b12 = cb2.req_const_block(i12);
            dense_tensor_wr_i<2, double> &c12 = cc.req_block(i12);
            tod_copy<2>(b12).perform(false, c12);
            out.put(i12, b12, tr0);
            cc.ret_block(i12);
            cb2.ret_const_block(i12);
        }

        out.close();
    }

    //  Compare against reference

    dense_tensor<2, double, allocator_type> tc(dims), tc_ref(dims);
    tod_btconv<2>(btc).perform(tc);
    tod_btconv<2>(btc_ref).perform(tc_ref);

    compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void gen_bto_aux_add_test::test_2() {

    static const char *testname = "gen_bto_aux_add_test::test_2()";

    typedef std_allocator<double> allocator_type;
    typedef block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i1, i2;
    i2[0] = 11; i2[1] = 11;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    bis.split(m11, 2);
    bis.split(m11, 5);
    dimensions<2> bidims(bis.get_block_index_dims());

    symmetry<2, double> syma(bis), symb(bis), symc(bis);
    se_perm<2, double> se1(permutation<2>().permute(0, 1),
        scalar_transf<double>(1.0));
    syma.insert(se1);
    symb.insert(se1);

    index<2> i00, i12;
    i12[0] = 1; i12[1] = 2;

    block_tensor<2, double, allocator_type> bta(bis), btb(bis), btc(bis),
        btc_ref(bis);

    //  Set up initial A

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> ca(bta);
        so_copy<2, double>(syma).perform(ca.req_symmetry());
    }
    bta.set_immutable();

    //  Set up initial B

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> cb(btb);
        so_copy<2, double>(symb).perform(cb.req_symmetry());

        dense_tensor<2, double, allocator_type> t00(bis.get_block_dims(i00));
        tod_random<2>().perform(t00);

        dense_tensor_wr_i<2, double> &b00 = cb.req_block(i00);
        tod_copy<2>(t00, 0.5).perform(true, b00);
        tod_copy<2>(t00, permutation<2>().permute(0, 1), 0.5).
            perform(false, b00);
        cb.ret_block(i00);
        dense_tensor_wr_i<2, double> &b12 = cb.req_block(i12);
        tod_random<2>().perform(b12);
        cb.ret_block(i12);
    }
    btb.set_immutable();

    //  Build schedules

    assignment_schedule<2, double> sch(bidims);
    sch.insert(i00);
    sch.insert(i12);

    addition_schedule<2, btod_traits> asch(syma, symb);
    {
        gen_block_tensor_rd_ctrl<2, bti_traits> ca(bta);
        asch.build(sch, ca);
    }

    //  Send blocks to the addition stream

    gen_bto_aux_add<2, btod_traits> out(syma, asch, btc,
        scalar_transf<double>(1.0));
    {
        gen_block_tensor_rd_ctrl<2, bti_traits> cb(btb);
        gen_block_tensor_wr_ctrl<2, bti_traits> cc(btc_ref);
        so_copy<2, double>(symc).perform(cc.req_symmetry());
        tensor_transf<2, double> tr0;
        out.open();

        dense_tensor_rd_i<2, double> &b00 = cb.req_const_block(i00);
        dense_tensor_wr_i<2, double> &c00 = cc.req_block(i00);
        tod_copy<2>(b00).perform(true, c00);
        out.put(i00, b00, tr0);
        cc.ret_block(i00);
        cb.ret_const_block(i00);

        dense_tensor_rd_i<2, double> &b12 = cb.req_const_block(i12);
        dense_tensor_wr_i<2, double> &c12 = cc.req_block(i12);
        tod_copy<2>(b12).perform(true, c12);
        out.put(i12, b12, tr0);
        cc.ret_block(i12);
        cb.ret_const_block(i12);

        out.close();
    }

    //  Compare against reference

    dense_tensor<2, double, allocator_type> tc(dims), tc_ref(dims);
    tod_btconv<2>(btc).perform(tc);
    tod_btconv<2>(btc_ref).perform(tc_ref);

    compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void gen_bto_aux_add_test::test_3a() {

    static const char *testname = "gen_bto_aux_add_test::test_3a()";

    typedef std_allocator<double> allocator_type;
    typedef block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i1, i2;
    i2[0] = 11; i2[1] = 11;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    bis.split(m11, 2);
    bis.split(m11, 5);
    dimensions<2> bidims(bis.get_block_index_dims());

    symmetry<2, double> syma(bis), symb(bis), symc(bis);
    se_perm<2, double> se1(permutation<2>().permute(0, 1),
        scalar_transf<double>(1.0));
    se_perm<2, double> se2(permutation<2>().permute(0, 1),
        scalar_transf<double>(-1.0));
    syma.insert(se1);
    symb.insert(se2);

    index<2> i00, i11, i12, i21;
    i11[0] = 1; i11[1] = 1;
    i12[0] = 1; i12[1] = 2;
    i21[0] = 2; i21[1] = 1;

    block_tensor<2, double, allocator_type> bta(bis), btb(bis), btc(bis),
        btc_ref(bis);

    //  Set up initial A

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> ca(bta);
        so_copy<2, double>(syma).perform(ca.req_symmetry());

        dense_tensor<2, double, allocator_type> t11(bis.get_block_dims(i11));
        tod_random<2>().perform(t11);

        dense_tensor_wr_i<2, double> &a11 = ca.req_block(i11);
        tod_copy<2>(t11, 0.5).perform(true, a11);
        tod_copy<2>(t11, permutation<2>().permute(0, 1), 0.5).
            perform(false, a11);
        ca.ret_block(i11);

        dense_tensor_wr_i<2, double> &a12 = ca.req_block(i12);
        tod_random<2>().perform(a12);
        ca.ret_block(i12);
    }
    bta.set_immutable();

    //  Set up initial B

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> cb(btb);
        so_copy<2, double>(symb).perform(cb.req_symmetry());

        dense_tensor<2, double, allocator_type> t00(bis.get_block_dims(i00));
        tod_random<2>().perform(t00);

        dense_tensor_wr_i<2, double> &b00 = cb.req_block(i00);
        tod_copy<2>(t00, 0.5).perform(true, b00);
        tod_copy<2>(t00, permutation<2>().permute(0, 1), -0.5).
            perform(false, b00);
        cb.ret_block(i00);
        dense_tensor_wr_i<2, double> &b12 = cb.req_block(i12);
        tod_random<2>().perform(b12);
        cb.ret_block(i12);
    }
    btb.set_immutable();

    //  Set up initial C (equal to A)

    {
        gen_block_tensor_rd_ctrl<2, bti_traits> ca(bta);
        gen_block_tensor_wr_ctrl<2, bti_traits> cc(btc);
        so_copy<2, double>(syma).perform(cc.req_symmetry());

        dense_tensor_rd_i<2, double> &a11 = ca.req_const_block(i11);
        dense_tensor_wr_i<2, double> &c11 = cc.req_block(i11);
        tod_copy<2>(a11).perform(true, c11);
        cc.ret_block(i11);
        ca.ret_const_block(i11);

        dense_tensor_rd_i<2, double> &a12 = ca.req_const_block(i12);
        dense_tensor_wr_i<2, double> &c12 = cc.req_block(i12);
        tod_copy<2>(a12).perform(true, c12);
        cc.ret_block(i12);
        ca.ret_const_block(i12);
    }

    //  Build schedules

    assignment_schedule<2, double> sch(bidims);
    sch.insert(i00);
    sch.insert(i12);

    addition_schedule<2, btod_traits> asch(syma, symb);
    {
        gen_block_tensor_rd_ctrl<2, bti_traits> ca(bta);
        asch.build(sch, ca);
    }

    //  Send blocks to the addition stream

    gen_bto_aux_add<2, btod_traits> out(syma, asch, btc,
        scalar_transf<double>(1.0));
    {
        gen_block_tensor_rd_ctrl<2, bti_traits> ca(bta);
        gen_block_tensor_rd_ctrl<2, bti_traits> cb(btb);
        gen_block_tensor_wr_ctrl<2, bti_traits> cc(btc_ref);
        so_copy<2, double>(symc).perform(cc.req_symmetry());
        tensor_transf<2, double> tr0;
        out.open();

        {
            dense_tensor_rd_i<2, double> &a11 = ca.req_const_block(i11);
            dense_tensor_wr_i<2, double> &c11 = cc.req_block(i11);
            tod_copy<2>(a11).perform(true, c11);
            cc.ret_block(i11);
            ca.ret_const_block(i11);
        }

        {
            dense_tensor_rd_i<2, double> &a12 = ca.req_const_block(i12);
            dense_tensor_wr_i<2, double> &c12 = cc.req_block(i12);
            dense_tensor_wr_i<2, double> &c21 = cc.req_block(i21);
            tod_copy<2>(a12).perform(true, c12);
            tod_copy<2>(a12, permutation<2>().permute(0, 1), 1.0).
                perform(true, c21);
            cc.ret_block(i12);
            cc.ret_block(i21);
            ca.ret_const_block(i12);
        }

        {
            dense_tensor_rd_i<2, double> &b00 = cb.req_const_block(i00);
            dense_tensor_wr_i<2, double> &c00 = cc.req_block(i00);
            tod_copy<2>(b00).perform(true, c00);
            out.put(i00, b00, tr0);
            cc.ret_block(i00);
            cb.ret_const_block(i00);
        }

        {
            dense_tensor_rd_i<2, double> &b12 = cb.req_const_block(i12);
            dense_tensor_wr_i<2, double> &c12 = cc.req_block(i12);
            dense_tensor_wr_i<2, double> &c21 = cc.req_block(i21);
            tod_copy<2>(b12).perform(false, c12);
            tod_copy<2>(b12, permutation<2>().permute(0, 1), -1.0).
                perform(false, c21);
            out.put(i12, b12, tr0);
            cc.ret_block(i12);
            cc.ret_block(i21);
            cb.ret_const_block(i12);
        }

        out.close();
    }

    //  Compare against reference

    dense_tensor<2, double, allocator_type> tc(dims), tc_ref(dims);
    tod_btconv<2>(btc).perform(tc);
    tod_btconv<2>(btc_ref).perform(tc_ref);

    compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void gen_bto_aux_add_test::test_3b() {

    static const char *testname = "gen_bto_aux_add_test::test_3b()";

    typedef std_allocator<double> allocator_type;
    typedef block_tensor_i_traits<double> bti_traits;

    try {

    index<2> i1, i2;
    i2[0] = 11; i2[1] = 11;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m11;
    m11[0] = true; m11[1] = true;
    bis.split(m11, 2);
    bis.split(m11, 5);
    dimensions<2> bidims(bis.get_block_index_dims());

    symmetry<2, double> syma(bis), symb(bis), symc(bis);
    se_perm<2, double> se1(permutation<2>().permute(0, 1),
        scalar_transf<double>(1.0));
    se_perm<2, double> se2(permutation<2>().permute(0, 1),
        scalar_transf<double>(-1.0));
    syma.insert(se1);
    symb.insert(se2);

    index<2> i00, i11, i12, i21;
    i11[0] = 1; i11[1] = 1;
    i12[0] = 1; i12[1] = 2;
    i21[0] = 2; i21[1] = 1;

    block_tensor<2, double, allocator_type> bta(bis), btb1(bis), btb2(bis),
        btc(bis), btc_ref(bis);

    //  Set up initial A

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> ca(bta);
        so_copy<2, double>(syma).perform(ca.req_symmetry());

        dense_tensor<2, double, allocator_type> t11(bis.get_block_dims(i11));
        tod_random<2>().perform(t11);

        dense_tensor_wr_i<2, double> &a11 = ca.req_block(i11);
        tod_copy<2>(t11, 0.5).perform(true, a11);
        tod_copy<2>(t11, permutation<2>().permute(0, 1), 0.5).
            perform(false, a11);
        ca.ret_block(i11);

        dense_tensor_wr_i<2, double> &a12 = ca.req_block(i12);
        tod_random<2>().perform(a12);
        ca.ret_block(i12);
    }
    bta.set_immutable();

    //  Set up initial B

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> cb1(btb1);
        gen_block_tensor_wr_ctrl<2, bti_traits> cb2(btb2);
        so_copy<2, double>(symb).perform(cb1.req_symmetry());
        so_copy<2, double>(symb).perform(cb2.req_symmetry());

        {
            dense_tensor<2, double, allocator_type> t00(
                bis.get_block_dims(i00));
            tod_random<2>().perform(t00);

            dense_tensor_wr_i<2, double> &b00 = cb1.req_block(i00);
            tod_copy<2>(t00, 0.5).perform(true, b00);
            tod_copy<2>(t00, permutation<2>().permute(0, 1), -0.5).
                perform(false, b00);
            cb1.ret_block(i00);
        }

        {
            dense_tensor_wr_i<2, double> &b12 = cb1.req_block(i12);
            tod_random<2>().perform(b12);
            cb1.ret_block(i12);
        }

        {
            dense_tensor_wr_i<2, double> &b12 = cb2.req_block(i12);
            tod_random<2>().perform(b12);
            cb2.ret_block(i12);
        }
    }
    btb1.set_immutable();
    btb2.set_immutable();

    //  Set up initial C (equal to A)

    {
        gen_block_tensor_rd_ctrl<2, bti_traits> ca(bta);
        gen_block_tensor_wr_ctrl<2, bti_traits> cc(btc);
        so_copy<2, double>(syma).perform(cc.req_symmetry());

        dense_tensor_rd_i<2, double> &a11 = ca.req_const_block(i11);
        dense_tensor_wr_i<2, double> &c11 = cc.req_block(i11);
        tod_copy<2>(a11).perform(true, c11);
        cc.ret_block(i11);
        ca.ret_const_block(i11);

        dense_tensor_rd_i<2, double> &a12 = ca.req_const_block(i12);
        dense_tensor_wr_i<2, double> &c12 = cc.req_block(i12);
        tod_copy<2>(a12).perform(true, c12);
        cc.ret_block(i12);
        ca.ret_const_block(i12);
    }

    //  Build schedules

    assignment_schedule<2, double> sch(bidims);
    sch.insert(i00);
    sch.insert(i12);

    addition_schedule<2, btod_traits> asch(syma, symb);
    {
        gen_block_tensor_rd_ctrl<2, bti_traits> ca(bta);
        asch.build(sch, ca);
    }

    //  Send blocks to the addition stream

    gen_bto_aux_add<2, btod_traits> out(syma, asch, btc,
        scalar_transf<double>(1.0));
    {
        gen_block_tensor_rd_ctrl<2, bti_traits> ca(bta);
        gen_block_tensor_rd_ctrl<2, bti_traits> cb1(btb1);
        gen_block_tensor_rd_ctrl<2, bti_traits> cb2(btb2);
        gen_block_tensor_wr_ctrl<2, bti_traits> cc(btc_ref);
        so_copy<2, double>(symc).perform(cc.req_symmetry());
        tensor_transf<2, double> tr0;
        out.open();

        {
            dense_tensor_rd_i<2, double> &a11 = ca.req_const_block(i11);
            dense_tensor_wr_i<2, double> &c11 = cc.req_block(i11);
            tod_copy<2>(a11).perform(true, c11);
            cc.ret_block(i11);
            ca.ret_const_block(i11);
        }

        {
            dense_tensor_rd_i<2, double> &a12 = ca.req_const_block(i12);
            dense_tensor_wr_i<2, double> &c12 = cc.req_block(i12);
            dense_tensor_wr_i<2, double> &c21 = cc.req_block(i21);
            tod_copy<2>(a12).perform(true, c12);
            tod_copy<2>(a12, permutation<2>().permute(0, 1), 1.0).
                perform(true, c21);
            cc.ret_block(i12);
            cc.ret_block(i21);
            ca.ret_const_block(i12);
        }

        {
            dense_tensor_rd_i<2, double> &b00 = cb1.req_const_block(i00);
            dense_tensor_wr_i<2, double> &c00 = cc.req_block(i00);
            tod_copy<2>(b00).perform(true, c00);
            out.put(i00, b00, tr0);
            cc.ret_block(i00);
            cb1.ret_const_block(i00);
        }

        {
            dense_tensor_rd_i<2, double> &b12 = cb1.req_const_block(i12);
            dense_tensor_wr_i<2, double> &c12 = cc.req_block(i12);
            dense_tensor_wr_i<2, double> &c21 = cc.req_block(i21);
            tod_copy<2>(b12).perform(false, c12);
            tod_copy<2>(b12, permutation<2>().permute(0, 1), -1.0).
                perform(false, c21);
            out.put(i12, b12, tr0);
            cc.ret_block(i12);
            cc.ret_block(i21);
            cb1.ret_const_block(i12);
        }

        {
            dense_tensor_rd_i<2, double> &b12 = cb2.req_const_block(i12);
            dense_tensor_wr_i<2, double> &c12 = cc.req_block(i12);
            dense_tensor_wr_i<2, double> &c21 = cc.req_block(i21);
            tod_copy<2>(b12).perform(false, c12);
            tod_copy<2>(b12, permutation<2>().permute(0, 1), -1.0).
                perform(false, c21);
            out.put(i12, b12, tr0);
            cc.ret_block(i12);
            cc.ret_block(i21);
            cb2.ret_const_block(i12);
        }

        out.close();
    }

    //  Compare against reference

    dense_tensor<2, double, allocator_type> tc(dims), tc_ref(dims);
    tod_btconv<2>(btc).perform(tc);
    tod_btconv<2>(btc_ref).perform(tc_ref);

    compare_ref<2>::compare(testname, tc, tc_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


} // namespace libtensor
