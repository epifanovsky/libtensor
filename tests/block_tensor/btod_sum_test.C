#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/btod_add.h>
#include <libtensor/block_tensor/btod_contract2.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/block_tensor/btod_sum.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include "btod_sum_test.h"
#include "../compare_ref.h"

namespace libtensor {

void btod_sum_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(4, 16, 16777216, 16777216);

    try {

    test_1();
    test_2();
    test_3();
    test_4();
    test_5();
    test_6(true);
    test_6(false);
    test_7();
    test_8();
    test_9a();
    test_9b();
    test_10a();
    test_10b();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void btod_sum_test::test_1() throw(libtest::test_exception) {

    //
    //  Single operand A + B
    //

    static const char *testname = "btod_sum_test::test_1()";

    typedef allocator<double> allocator_t;
    typedef block_tensor<2, double, allocator_t> block_tensor_t;

    try {

    index<2> i1, i2;
    i2[0] = 5; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    block_tensor_t bt1(bis), bt2(bis), bt3(bis), bt3_ref(bis);
    btod_random<2>().perform(bt1);
    btod_random<2>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();


    btod_add<2> add(bt1);
    add.add_op(bt2, 2.0);
    add.add_op(bt2);

    btod_sum<2> sum(add);
    sum.perform(bt3);
    add.perform(bt3_ref);

    compare_ref<2>::compare(testname, bt3, bt3_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_sum_test::test_2() throw(libtest::test_exception) {

    //
    //  Two operands: A + B and C + D
    //

    static const char *testname = "btod_sum_test::test_2()";

    typedef allocator<double> allocator_t;
    typedef block_tensor<2, double, allocator_t> block_tensor_t;

    try {

    index<2> i1, i2;
    i2[0] = 5; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    block_tensor_t bt1(bis), bt2(bis), bt3(bis), bt4(bis),
        bt5(bis), bt5_ref(bis);
    btod_random<2>().perform(bt1);
    btod_random<2>().perform(bt2);
    btod_random<2>().perform(bt3);
    btod_random<2>().perform(bt4);
    btod_random<2>().perform(bt5);
    btod_copy<2>(bt5).perform(bt5_ref);
    bt1.set_immutable();
    bt2.set_immutable();
    bt3.set_immutable();
    bt4.set_immutable();

    btod_add<2> add1(bt1), add2(bt3), add_ref(bt1);
    add1.add_op(bt2);
    add2.add_op(bt4);
    add_ref.add_op(bt2);
    add_ref.add_op(bt3);
    add_ref.add_op(bt4);

    btod_sum<2> sum(add1);
    sum.add_op(add2);
    sum.perform(bt5);
    add_ref.perform(bt5_ref);

    compare_ref<2>::compare(testname, bt5, bt5_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_sum_test::test_3() throw(libtest::test_exception) {

    //
    //  Two operands: A + B and C + D
    //

    static const char *testname = "btod_sum_test::test_3()";

    typedef allocator<double> allocator_t;
    typedef block_tensor<2, double, allocator_t> block_tensor_t;

    try {

    index<2> i1, i2;
    i2[0] = 5; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);

    block_tensor_t bt1(bis), bt2(bis), bt3(bis), bt4(bis),
        bt5(bis), bt5_ref(bis);
    btod_random<2>().perform(bt1);
    btod_random<2>().perform(bt2);
    btod_random<2>().perform(bt3);
    btod_random<2>().perform(bt4);
    bt1.set_immutable();
    bt2.set_immutable();
    bt3.set_immutable();
    bt4.set_immutable();

    btod_add<2> add1(bt1), add2(bt3), add_ref(bt1);
    add1.add_op(bt2);
    add2.add_op(bt4);
    add_ref.add_op(bt2);
    add_ref.add_op(bt3, -1.0);
    add_ref.add_op(bt4, -1.0);

    btod_sum<2> sum(add1);
    sum.add_op(add2, -1.0);
    sum.perform(bt5);
    add_ref.perform(bt5_ref);

    compare_ref<2>::compare(testname, bt5, bt5_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_sum_test::test_4() throw(libtest::test_exception) {

    //
    //  Two operands: A and C + D
    //

    static const char *testname = "btod_sum_test::test_4()";

    typedef allocator<double> allocator_t;
    typedef block_tensor<4, double, allocator_t> block_tensor_t;

    try {

    index<4> i1, i2;
    i2[0] = 5; i2[1] = 10; i2[2] = 5; i2[3] = 10;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1, m2;
    m1[0] = true; m1[2] = true;
    m2[1] = true; m2[3] = true;
    bis.split(m1, 2);
    bis.split(m2, 3);
    bis.split(m2, 6);

    block_tensor_t bt1(bis), bt2(bis), bt3(bis), bt4(bis), bt4_ref(bis);
    btod_random<4>().perform(bt1);
    btod_random<4>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    permutation<4> perm;
    perm.permute(1, 3);
    btod_add<4> add1(bt1), add2(bt2), add_ref(bt1);
    add2.add_op(bt2, perm, -1.0);
    btod_copy<4>(bt2, perm, -1.0).perform(bt3);
    bt3.set_immutable();
    add_ref.add_op(bt2);
    add_ref.add_op(bt3);

    btod_sum<4> sum(add1);
    sum.add_op(add2);
    sum.perform(bt4);
    add_ref.perform(bt4_ref);

    compare_ref<4>::compare(testname, bt4, bt4_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_sum_test::test_5() throw(libtest::test_exception) {

    //
    //  Single operand A * B
    //

    static const char *testname = "btod_sum_test::test_5()";

    typedef allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 12; i2[1] = 12; i2[2] = 6; i2[3] = 6;
    dimensions<4> dims_iiaa(index_range<4>(i1, i2));
    i2[0] = 12; i2[1] = 6; i2[2] = 6; i2[3] = 6;
    dimensions<4> dims_iaaa(index_range<4>(i1, i2));
    block_index_space<4> bis_iiaa(dims_iiaa), bis_iaaa(dims_iaaa);
    mask<4> m1, m2, m3, m4;
    m1[0] = true; m1[1] = true; m2[2] = true; m2[3] = true;
    m3[0] = true; m4[1] = true; m4[2] = true; m4[3] = true;
    bis_iiaa.split(m1, 3);
    bis_iiaa.split(m1, 7);
    bis_iiaa.split(m1, 10);
    bis_iiaa.split(m2, 2);
    bis_iiaa.split(m2, 3);
    bis_iiaa.split(m2, 5);
    bis_iaaa.split(m3, 3);
    bis_iaaa.split(m3, 7);
    bis_iaaa.split(m3, 10);
    bis_iaaa.split(m4, 2);
    bis_iaaa.split(m4, 3);
    bis_iaaa.split(m4, 5);

    block_tensor<4, double, allocator_t> bta(bis_iaaa);
    block_tensor<4, double, allocator_t> btb(bis_iiaa);
    block_tensor<4, double, allocator_t> btc(bis_iaaa), btc_ref(bis_iaaa);

    //  Load random data for input

    btod_random<4>().perform(bta);
    btod_random<4>().perform(btb);
    bta.set_immutable();
    btb.set_immutable();

    //  Run contraction and compute the reference

    //  iabc = kcad ikbd
    //  caib->iabc
    contraction2<2, 2, 2> contr(permutation<4>().permute(0, 2).
        permute(2, 3));
    contr.contract(0, 1);
    contr.contract(3, 3);

    btod_contract2<2, 2, 2> op(contr, bta, btb);
    op.perform(btc_ref);

    btod_sum<4> sum(op);
    sum.perform(btc);

    compare_ref<4>::compare(testname, btc, btc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void btod_sum_test::test_6(bool do_add) throw(libtest::test_exception) {

    //
    //  Two operands A + B and C + D, symmetry
    //

    std::ostringstream tnss;
    tnss << "btod_sum_test::test_6(" << do_add << ")";

    typedef allocator<double> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 12; i2[1] = 12; i2[2] = 6; i2[3] = 6;
    dimensions<4> dims_iiaa(index_range<4>(i1, i2));
    i2[0] = 12; i2[1] = 6; i2[2] = 6; i2[3] = 6;
    block_index_space<4> bis_iiaa(dims_iiaa);
    mask<4> m1, m2;
    m1[0] = true; m1[1] = true; m2[2] = true; m2[3] = true;
    bis_iiaa.split(m1, 3);
    bis_iiaa.split(m1, 7);
    bis_iiaa.split(m1, 10);
    bis_iiaa.split(m2, 2);
    bis_iiaa.split(m2, 3);
    bis_iiaa.split(m2, 5);

    block_tensor<4, double, allocator_t> bta1(bis_iiaa), bta2(bis_iiaa);
    block_tensor<4, double, allocator_t> btb1(bis_iiaa), btb2(bis_iiaa);
    block_tensor<4, double, allocator_t> btc(bis_iiaa), btc_ref(bis_iiaa);

    {
    block_tensor_ctrl<4, double> ctrl_a1(bta1), ctrl_a2(bta2);
    block_tensor_ctrl<4, double> ctrl_b1(btb1), ctrl_b2(btb2);
    block_tensor_ctrl<4, double> ctrl_c(btc), ctrl_c_ref(btc_ref);
    permutation<4> p1023, p0132;
    p1023.permute(0, 1);
    p0132.permute(2, 3);
    scalar_transf<double> tr0, tr1(-1.);
    se_perm<4, double> sp1023(p1023, tr0), sp0132(p0132, tr1);
    ctrl_a1.req_symmetry().insert(sp1023);
    ctrl_a1.req_symmetry().insert(sp0132);
    ctrl_b1.req_symmetry().insert(sp1023);
    ctrl_b1.req_symmetry().insert(sp0132);
    ctrl_a2.req_symmetry().insert(sp1023);
    ctrl_b2.req_symmetry().insert(sp1023);
    ctrl_c.req_symmetry().insert(sp1023);
    ctrl_c.req_symmetry().insert(sp0132);
    ctrl_c_ref.req_symmetry().insert(sp1023);
    ctrl_c_ref.req_symmetry().insert(sp0132);
    }

    //  Load random data for input

    btod_random<4>().perform(bta1);
    btod_random<4>().perform(btb1);
    btod_random<4>().perform(bta2);
    btod_random<4>().perform(btb2);
    bta1.set_immutable();
    btb1.set_immutable();
    bta2.set_immutable();
    btb2.set_immutable();

    //  Prepare reference

    if(do_add) {
        btod_random<4>().perform(btc);
        btod_copy<4>(btc).perform(btc_ref);
    }

    //  Run contraction and compute the reference

    btod_add<4> op1(bta1), op2(bta2);
    op1.add_op(btb1);
    op2.add_op(btb2);
    if(do_add) op1.perform(btc_ref, 1.0);
    else op1.perform(btc_ref);
    op2.perform(btc_ref, 1.0);

    btod_sum<4> sum(op1);
    sum.add_op(op2);
    if(do_add) sum.perform(btc, 1.0);
    else sum.perform(btc);

    compare_ref<4>::compare(tnss.str().c_str(), btc, btc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


void btod_sum_test::test_7() throw(libtest::test_exception) {

    //
    //  Single operand A + B, permutational symmetry
    //

    static const char *testname = "btod_sum_test::test_7()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 10; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m;
    m[0] = true; m[1] = true;
    bis.split(m, 4);
    scalar_transf<double> tr0, tr1(-1.);

    block_tensor<2, double, allocator_t> bt1(bis), bt2(bis), bt3(bis),
        bt3_ref(bis);
    {
        block_tensor_ctrl<2, double> ctrl1(bt1), ctrl2(bt2);
        ctrl1.req_symmetry().insert(se_perm<2, double>(
            permutation<2>().permute(0, 1), tr0));
        ctrl2.req_symmetry().insert(se_perm<2, double>(
            permutation<2>().permute(0, 1), tr0));
    }
    btod_random<2>().perform(bt1);
    btod_random<2>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    btod_add<2> add(bt1);
    add.add_op(bt2, 2.0);

    btod_sum<2> sum(add);
    sum.perform(bt3);
    add.perform(bt3_ref);

    symmetry<2, double> sym3(bis), sym3_ref(bis);
    {
        block_tensor_ctrl<2, double> ctrl3(bt3);
        so_copy<2, double>(ctrl3.req_const_symmetry()).perform(sym3);
        sym3_ref.insert(se_perm<2, double>(
            permutation<2>().permute(0, 1), tr0));
    }

    compare_ref<2>::compare(testname, sym3, sym3_ref);
    compare_ref<2>::compare(testname, bt3, bt3_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_sum_test::test_8() throw(libtest::test_exception) {

    //
    //  Two operands with different permutational symmetry
    //

    static const char *testname = "btod_sum_test::test_8()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 10; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m;
    m[0] = true; m[1] = true;
    bis.split(m, 4);

    block_tensor<2, double, allocator_t> bt1(bis), bt2(bis), bt3(bis),
        bt3_ref(bis);
    {
        block_tensor_ctrl<2, double> ctrl1(bt1), ctrl2(bt2);
        scalar_transf<double> tr0, tr1(-1.);
        ctrl1.req_symmetry().insert(se_perm<2, double>(
            permutation<2>().permute(0, 1), tr0));
    }
    btod_random<2>().perform(bt1);
    btod_random<2>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    btod_copy<2> cp1(bt1);
    btod_copy<2> cp2(bt2);
    cp1.perform(bt3_ref);
    cp2.perform(bt3_ref, 1.0);

    btod_sum<2> sum(cp1);
    sum.add_op(cp2);
    sum.perform(bt3);

    symmetry<2, double> sym3(bis), sym3_ref(bis);
    {
        block_tensor_ctrl<2, double> ctrl3(bt3);
        so_copy<2, double>(ctrl3.req_const_symmetry()).perform(sym3);
    }

    compare_ref<2>::compare(testname, sym3, sym3_ref);
    compare_ref<2>::compare(testname, bt3, bt3_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_sum_test::test_9a() throw(libtest::test_exception) {

    //
    //  tt_oovv(i|j|a|b) = t2(i|j|a|b) - t1(j|a)*t1(i|b);
    //

    static const char *testname = "btod_sum_test::test_9a()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i2a, i2b;
    i2b[0] = 5; i2b[1] = 10;
    dimensions<2> dims_ia(index_range<2>(i2a, i2b));
    index<4> i4a, i4b;
    i4b[0] = 5; i4b[1] = 5; i4b[2] = 10; i4b[3] = 10;
    dimensions<4> dims_ijab(index_range<4>(i4a, i4b));

    block_index_space<2> bis_ia(dims_ia);
    block_index_space<4> bis_ijab(dims_ijab);
    mask<2> m2a, m2b;
    m2a[0] = true; m2b[1] = true;
    bis_ia.split(m2a, 2);
    bis_ia.split(m2b, 6);
    mask<4> m4a, m4b;
    m4a[0] = true; m4a[1] = true; m4b[2] = true; m4b[3] = true;
    bis_ijab.split(m4a, 2);
    bis_ijab.split(m4b, 6);

    block_tensor<2, double, allocator_t> bt1(bis_ia);
    block_tensor<4, double, allocator_t> bt2(bis_ijab), bt3(bis_ijab),
        bt3_ref(bis_ijab);
    {
        block_tensor_ctrl<4, double> ctrl2(bt2);
        scalar_transf<double> tr0, tr1(-1.);
        ctrl2.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr1));
        ctrl2.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(2, 3), tr1));
    }
    btod_random<2>().perform(bt1);
    btod_random<4>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    btod_copy<4> op1(bt2);
    contraction2<2, 2, 0> contr1(permutation<4>().
        permute(1, 2).permute(0, 1));
    btod_contract2<2, 2, 0> op2(contr1, bt1, bt1);

    op1.perform(bt3_ref);
    op2.perform(bt3_ref, -1.0);

    btod_sum<4> sum(op1);
    sum.add_op(op2, -1.0);
    sum.perform(bt3);

    compare_ref<4>::compare(testname, bt3, bt3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_sum_test::test_9b() throw(libtest::test_exception) {

    //
    //  tt_oovv(i|j|a|b) = t2(i|j|a|b) - t1(j|a)*t1(i|b);
    //  performed blockwise
    //

    static const char *testname = "btod_sum_test::test_9b()";

    typedef allocator<double> allocator_t;

    try {

    index<2> i2a, i2b;
    i2b[0] = 5; i2b[1] = 10;
    dimensions<2> dims_ia(index_range<2>(i2a, i2b));
    index<4> i4a, i4b;
    i4b[0] = 5; i4b[1] = 5; i4b[2] = 10; i4b[3] = 10;
    dimensions<4> dims_ijab(index_range<4>(i4a, i4b));

    block_index_space<2> bis_ia(dims_ia);
    block_index_space<4> bis_ijab(dims_ijab);
    mask<2> m2a, m2b;
    m2a[0] = true; m2b[1] = true;
    bis_ia.split(m2a, 2);
    bis_ia.split(m2b, 6);
    mask<4> m4a, m4b;
    m4a[0] = true; m4a[1] = true; m4b[2] = true; m4b[3] = true;
    bis_ijab.split(m4a, 2);
    bis_ijab.split(m4b, 6);

    block_tensor<2, double, allocator_t> bt1(bis_ia);
    block_tensor<4, double, allocator_t> bt2(bis_ijab), bt3(bis_ijab),
        bt3_ref(bis_ijab);
    {
        block_tensor_ctrl<4, double> ctrl2(bt2);
        scalar_transf<double> tr0, tr1(-1.);
        ctrl2.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr1));
        ctrl2.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(2, 3), tr1));
    }
    btod_random<2>().perform(bt1);
    btod_random<4>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    btod_copy<4> op1(bt2);
    contraction2<2, 2, 0> contr1(permutation<4>().
        permute(1, 2).permute(0, 1));
    btod_contract2<2, 2, 0> op2(contr1, bt1, bt1);

    op1.perform(bt3_ref);
    op2.perform(bt3_ref, -1.0);

    btod_sum<4> sum(op1);
    sum.add_op(op2, -1.0);

    const assignment_schedule<4, double> &sch = sum.get_schedule();
    block_tensor_ctrl<4, double> c3(bt3);
    so_copy<4, double>(sum.get_symmetry()).perform(c3.req_symmetry());
    for(assignment_schedule<4, double>::iterator i = sch.begin();
        i != sch.end(); i++) {

        abs_index<4> ijab(sch.get_abs_index(i),
            bis_ijab.get_block_index_dims());
        dense_tensor_wr_i<4, double> &blk = c3.req_block(ijab.get_index());
        sum.compute_block(true, ijab.get_index(),
                tensor_transf<4, double>(), blk);
        c3.ret_block(ijab.get_index());
    }

    compare_ref<4>::compare(testname, bt3, bt3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_sum_test::test_10a() throw(libtest::test_exception) {

    //
    //  Two operands: A and B, uneven block index space splits
    //

    static const char *testname = "btod_sum_test::test_10a()";

    typedef allocator<double> allocator_t;
    typedef block_tensor<2, double, allocator_t> block_tensor_t;

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis1(dims), bis2(dims);
    mask<2> m01, m10, m11;
    m10[0] = true; m01[1] = true;
    m11[0] = true; m11[1] = true;
    bis1.split(m01, 5);
    bis1.split(m10, 5);
    bis2.split(m11, 5);

    block_tensor_t bt1(bis1), bt2(bis2), bt3(bis2), bt3_ref(bis2);
    btod_random<2>().perform(bt1);
    btod_random<2>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    btod_copy<2> cp1(bt1), cp2(bt2);

    btod_sum<2> sum(cp1);
    sum.add_op(cp2);
    sum.perform(bt3);

    btod_copy<2>(bt1).perform(bt3_ref);
    btod_copy<2>(bt2).perform(bt3_ref, 1.0);

    compare_ref<2>::compare(testname, bt3, bt3_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btod_sum_test::test_10b() throw(libtest::test_exception) {

    //
    //  Two operands: A and B, uneven block index space splits
    //

    static const char *testname = "btod_sum_test::test_10b()";

    typedef allocator<double> allocator_t;
    typedef block_tensor<2, double, allocator_t> block_tensor_t;

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis1(dims), bis2(dims);
    mask<2> m01, m10, m11;
    m10[0] = true; m01[1] = true;
    m11[0] = true; m11[1] = true;
    bis1.split(m01, 5);
    bis1.split(m10, 5);
    bis2.split(m11, 5);

    block_tensor_t bt1(bis1), bt2(bis2), bt3(bis2), bt3_ref(bis2);
    btod_random<2>().perform(bt1);
    btod_random<2>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    permutation<2> p10; p10.permute(0, 1);

    btod_copy<2> cp1(bt1), cp2(bt2, p10);

    btod_sum<2> sum(cp1);
    sum.add_op(cp2);
    sum.perform(bt3);

    btod_copy<2>(bt1).perform(bt3_ref);
    btod_copy<2>(bt2, p10).perform(bt3_ref, 1.0);

    compare_ref<2>::compare(testname, bt3, bt3_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

