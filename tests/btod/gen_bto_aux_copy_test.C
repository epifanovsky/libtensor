#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include "../compare_ref.h"
#include "gen_bto_aux_copy_test.h"

namespace libtensor {


void gen_bto_aux_copy_test::perform() throw(libtest::test_exception) {

    allocator<double>::vmm().init(16, 16, 65536, 65536);

    try {

    test_1a();
    test_1b();
    test_1c();
    test_2();
    test_exc_1();

    } catch (...) {
        allocator<double>::vmm().shutdown();
        throw;
    }
    allocator<double>::vmm().shutdown();
}


void gen_bto_aux_copy_test::test_1a() {

    static const char *testname = "gen_bto_aux_copy_test::test_1a()";

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

    block_tensor<2, double, allocator_type> bta(bis), btb(bis), btb_ref(bis);

    //  Set up initial A

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> ca(bta);
        so_copy<2, double>(syma).perform(ca.req_symmetry());

        {
            dense_tensor_wr_i<2, double> &a00 = ca.req_block(i00);
            tod_random<2>().perform(a00);
            ca.ret_block(i00);
        }
        {
            dense_tensor_wr_i<2, double> &a12 = ca.req_block(i12);
            tod_random<2>().perform(a12);
            ca.ret_block(i12);
        }
    }
    bta.set_immutable();

    //  Send blocks to the stream

    gen_bto_aux_copy<2, btod_traits> out(symb, btb);

    {
        gen_block_tensor_rd_ctrl<2, bti_traits> ca(bta);
        gen_block_tensor_wr_ctrl<2, bti_traits> cb(btb_ref);
        tensor_transf<2, double> tr0;
        out.open();

        {
            dense_tensor_rd_i<2, double> &a00 = ca.req_const_block(i00);
            dense_tensor_wr_i<2, double> &b00 = cb.req_block(i00);
            tod_copy<2>(a00).perform(true, b00);
            out.put(i00, a00, tr0);
            cb.ret_block(i00);
            ca.ret_const_block(i00);
        }

        {
            dense_tensor_rd_i<2, double> &a12 = ca.req_const_block(i12);
            dense_tensor_wr_i<2, double> &b12 = cb.req_block(i12);
            tod_copy<2>(a12).perform(true, b12);
            out.put(i12, a12, tr0);
            cb.ret_block(i12);
            ca.ret_const_block(i12);
        }

        out.close();
    }

    //  Compare against reference

    dense_tensor<2, double, allocator_type> tb(dims), tb_ref(dims);
    tod_btconv<2>(btb).perform(tb);
    tod_btconv<2>(btb_ref).perform(tb_ref);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void gen_bto_aux_copy_test::test_1b() {

    static const char *testname = "gen_bto_aux_copy_test::test_1b()";

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

    block_tensor<2, double, allocator_type> bta1(bis), bta2(bis), btb(bis),
        btb_ref(bis);

    //  Set up initial A

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> ca1(bta1);
        gen_block_tensor_wr_ctrl<2, bti_traits> ca2(bta2);
        so_copy<2, double>(syma).perform(ca1.req_symmetry());
        so_copy<2, double>(syma).perform(ca2.req_symmetry());

        {
            dense_tensor_wr_i<2, double> &a00 = ca1.req_block(i00);
            tod_random<2>().perform(a00);
            ca1.ret_block(i00);
        }
        {
            dense_tensor_wr_i<2, double> &a12 = ca1.req_block(i12);
            tod_random<2>().perform(a12);
            ca1.ret_block(i12);
        }
        {
            dense_tensor_wr_i<2, double> &a12 = ca2.req_block(i12);
            tod_random<2>().perform(a12);
            ca2.ret_block(i12);
        }
    }
    bta1.set_immutable();
    bta2.set_immutable();

    //  Send blocks to the stream

    gen_bto_aux_copy<2, btod_traits> out(symb, btb);

    {
        gen_block_tensor_rd_ctrl<2, bti_traits> ca1(bta1);
        gen_block_tensor_rd_ctrl<2, bti_traits> ca2(bta2);
        gen_block_tensor_wr_ctrl<2, bti_traits> cb(btb_ref);
        tensor_transf<2, double> tr0;
        out.open();

        {
            dense_tensor_rd_i<2, double> &a00 = ca1.req_const_block(i00);
            dense_tensor_wr_i<2, double> &b00 = cb.req_block(i00);
            tod_copy<2>(a00).perform(true, b00);
            out.put(i00, a00, tr0);
            cb.ret_block(i00);
            ca1.ret_const_block(i00);
        }

        {
            dense_tensor_rd_i<2, double> &a12 = ca1.req_const_block(i12);
            dense_tensor_wr_i<2, double> &b12 = cb.req_block(i12);
            tod_copy<2>(a12).perform(true, b12);
            out.put(i12, a12, tr0);
            cb.ret_block(i12);
            ca1.ret_const_block(i12);
        }

        {
            dense_tensor_rd_i<2, double> &a12 = ca2.req_const_block(i12);
            dense_tensor_wr_i<2, double> &b12 = cb.req_block(i12);
            tod_copy<2>(a12).perform(false, b12);
            out.put(i12, a12, tr0);
            cb.ret_block(i12);
            ca2.ret_const_block(i12);
        }

        out.close();
    }

    //  Compare against reference

    dense_tensor<2, double, allocator_type> tb(dims), tb_ref(dims);
    tod_btconv<2>(btb).perform(tb);
    tod_btconv<2>(btb_ref).perform(tb_ref);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void gen_bto_aux_copy_test::test_1c() {

    static const char *testname = "gen_bto_aux_copy_test::test_1c()";

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

    block_tensor<2, double, allocator_type> bta1(bis), bta2(bis), btb(bis),
        btb_ref(bis);

    //  Set up initial A

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> ca1(bta1);
        gen_block_tensor_wr_ctrl<2, bti_traits> ca2(bta2);
        so_copy<2, double>(syma).perform(ca1.req_symmetry());
        so_copy<2, double>(syma).perform(ca2.req_symmetry());

        {
            dense_tensor_wr_i<2, double> &a00 = ca1.req_block(i00);
            tod_random<2>().perform(a00);
            ca1.ret_block(i00);
        }
        {
            dense_tensor_wr_i<2, double> &a12 = ca1.req_block(i12);
            tod_random<2>().perform(a12);
            ca1.ret_block(i12);
        }
        {
            dense_tensor_wr_i<2, double> &a12 = ca2.req_block(i12);
            tod_random<2>().perform(a12);
            ca2.ret_block(i12);
        }
    }
    bta1.set_immutable();
    bta2.set_immutable();

    //  Send blocks to the stream

    gen_bto_aux_copy<2, btod_traits> out(symb, btb);

    {
        gen_block_tensor_rd_ctrl<2, bti_traits> ca1(bta1);
        gen_block_tensor_rd_ctrl<2, bti_traits> ca2(bta2);
        gen_block_tensor_wr_ctrl<2, bti_traits> cb(btb_ref);
        tensor_transf<2, double> tr0;
        out.open();

        {
            dense_tensor_rd_i<2, double> &a00 = ca1.req_const_block(i00);
            out.put(i00, a00, tr0);
            ca1.ret_const_block(i00);
        }

        {
            dense_tensor_rd_i<2, double> &a12 = ca1.req_const_block(i12);
            out.put(i12, a12, tr0);
            ca1.ret_const_block(i12);
        }

        out.close();
        //  This should reset the output tensor to zero
        out.open();

        {
            dense_tensor_rd_i<2, double> &a12 = ca2.req_const_block(i12);
            dense_tensor_wr_i<2, double> &b12 = cb.req_block(i12);
            tod_copy<2>(a12).perform(true, b12);
            out.put(i12, a12, tr0);
            cb.ret_block(i12);
            ca2.ret_const_block(i12);
        }

        out.close();
    }

    //  Compare against reference

    dense_tensor<2, double, allocator_type> tb(dims), tb_ref(dims);
    tod_btconv<2>(btb).perform(tb);
    tod_btconv<2>(btb_ref).perform(tb_ref);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void gen_bto_aux_copy_test::test_2() {

    static const char *testname = "gen_bto_aux_copy_test::test_2()";

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
    se_perm<2, double> se1(permutation<2>().permute(0, 1),
        scalar_transf<double>(1.0));
    syma.insert(se1);
    symb.insert(se1);

    index<2> i00, i12;
    i12[0] = 1; i12[1] = 2;

    block_tensor<2, double, allocator_type> bta(bis), btb(bis), btb_ref(bis);

    //  Set up initial A

    {
        gen_block_tensor_wr_ctrl<2, bti_traits> ca(bta);
        so_copy<2, double>(syma).perform(ca.req_symmetry());

        {
            dense_tensor<2, double, allocator_type> t00(
                bis.get_block_dims(i00));
            tod_random<2>().perform(t00);
            dense_tensor_wr_i<2, double> &a00 = ca.req_block(i00);
            tod_copy<2>(t00, 0.5).perform(true, a00);
            tod_copy<2>(t00, permutation<2>().permute(0, 1), 0.5).
                perform(false, a00);
            ca.ret_block(i00);
        }
        {
            dense_tensor_wr_i<2, double> &a12 = ca.req_block(i12);
            tod_random<2>().perform(a12);
            ca.ret_block(i12);
        }
    }
    bta.set_immutable();

    //  Send blocks to the stream

    gen_bto_aux_copy<2, btod_traits> out(symb, btb);

    {
        gen_block_tensor_rd_ctrl<2, bti_traits> ca(bta);
        gen_block_tensor_wr_ctrl<2, bti_traits> cb(btb_ref);
        so_copy<2, double>(symb).perform(cb.req_symmetry());
        tensor_transf<2, double> tr0;
        out.open();

        {
            dense_tensor_rd_i<2, double> &a00 = ca.req_const_block(i00);
            dense_tensor_wr_i<2, double> &b00 = cb.req_block(i00);
            tod_copy<2>(a00).perform(true, b00);
            out.put(i00, a00, tr0);
            cb.ret_block(i00);
            ca.ret_const_block(i00);
        }

        {
            dense_tensor_rd_i<2, double> &a12 = ca.req_const_block(i12);
            dense_tensor_wr_i<2, double> &b12 = cb.req_block(i12);
            tod_copy<2>(a12).perform(true, b12);
            out.put(i12, a12, tr0);
            cb.ret_block(i12);
            ca.ret_const_block(i12);
        }

        out.close();
    }

    //  Compare against reference

    dense_tensor<2, double, allocator_type> tb(dims), tb_ref(dims);
    tod_btconv<2>(btb).perform(tb);
    tod_btconv<2>(btb_ref).perform(tb_ref);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void gen_bto_aux_copy_test::test_exc_1() {

    static const char *testname = "gen_bto_aux_copy_test::test_exc_1()";

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

    symmetry<2, double> symb(bis);

    index<2> i00, i12;
    i12[0] = 1; i12[1] = 2;

    tensor_transf<2, double> tr0;
    dense_tensor<2, double, allocator_type> t00(bis.get_block_dims(i00));
    tod_random<2>().perform(t00);

    block_tensor<2, double, allocator_type> btb(bis);

    bool ok = false;
    gen_bto_aux_copy<2, btod_traits> out1(symb, btb);
    try {
        out1.put(i00, t00, tr0);
    } catch(block_stream_exception &e) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected a block_stream_exception.");
    }

    ok = false;
    gen_bto_aux_copy<2, btod_traits> out2(symb, btb);
    out2.open();
    try {
        out2.open();
    } catch(block_stream_exception &e) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected a block_stream_exception.");
    }

    ok = false;
    gen_bto_aux_copy<2, btod_traits> out3(symb, btb);
    try {
        out3.close();
    } catch(block_stream_exception &e) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected a block_stream_exception.");
    }

    ok = false;
    gen_bto_aux_copy<2, btod_traits> out4(symb, btb);
    out4.open();
    out4.close();
    try {
        out4.close();
    } catch(block_stream_exception &e) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected a block_stream_exception.");
    }

    ok = false;
    gen_bto_aux_copy<2, btod_traits> out5(symb, btb);
    out5.open();
    out5.put(i00, t00, tr0);
    out5.close();
    try {
        out5.close();
    } catch(block_stream_exception &e) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected a block_stream_exception.");
    }

    ok = false;
    gen_bto_aux_copy<2, btod_traits> out6(symb, btb);
    out6.open();
    out6.put(i00, t00, tr0);
    out6.close();
    try {
        out6.put(i00, t00, tr0);
    } catch(block_stream_exception &e) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected a block_stream_exception.");
    }

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


} // namespace libtensor
