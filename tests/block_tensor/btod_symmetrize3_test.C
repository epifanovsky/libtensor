#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/btod_add.h>
#include <libtensor/block_tensor/btod_contract2.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/block_tensor/btod_symmetrize3.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include "btod_symmetrize3_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_symmetrize3_test::perform() throw(libtest::test_exception) {

    allocator<double>::init();

    try {

        test_1();
        test_2();
        test_3();
        test_4();
        test_5();
        test_6();
        test_7();
        test_8a();
        test_8b();
        test_9();
        test_10();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


/** \test Symmetrization of a non-symmetric 3-index block %tensor
        over three indexes
 **/
void btod_symmetrize3_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize3_test::test_1()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<3> i1, i2;
    i2[0] = 10; i2[1] = 10; i2[2] = 10;
    dimensions<3> dims(index_range<3>(i1, i2));
    block_index_space<3> bis(dims);
    mask<3> m;
    m[0] = true; m[1] = true; m[2] = true;
    bis.split(m, 2);
    bis.split(m, 5);

    block_tensor<3, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

    //  Fill in random input

    btod_random<3>().perform(bta);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<3, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
    tod_btconv<3>(bta).perform(ta);
    tod_add<3> refop(ta);
    refop.add_op(ta, permutation<3>().permute(0, 1), 1.0);
    refop.add_op(ta, permutation<3>().permute(0, 2), 1.0);
    refop.add_op(ta, permutation<3>().permute(1, 2), 1.0);
    refop.add_op(ta, permutation<3>().permute(0, 1).permute(0, 2), 1.0);
    refop.add_op(ta, permutation<3>().permute(0, 1).permute(1, 2), 1.0);
    refop.perform(true, tb_ref);

    symmetry<3, double> symb(bis), symb_ref(bis);
    scalar_transf<double> tr0, tr1(-1.);
    symb_ref.insert(se_perm<3, double>(
        permutation<3>().permute(0, 1), tr0));
    symb_ref.insert(se_perm<3, double>(
        permutation<3>().permute(0, 2), tr0));

    //  Run the symmetrization operation

    btod_copy<3> op_copy(bta);
    btod_symmetrize3<3> op_sym(op_copy, 0, 1, 2, true);

    compare_ref<3>::compare(testname, op_sym.get_symmetry(), symb_ref);

    op_sym.perform(btb);
    tod_btconv<3>(btb).perform(tb);

    //  Compare against the reference: symmetry and data

    {
        block_tensor_ctrl<3, double> ctrlb(btb);
        so_copy<3, double>(ctrlb.req_const_symmetry()).perform(symb);
    }

    compare_ref<3>::compare(testname, symb, symb_ref);
    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


/** \test Anti-symmetrization of a non-symmetric 3-index block %tensor
        over three indexes
 **/
void btod_symmetrize3_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize3_test::test_2()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<3> i1, i2;
    i2[0] = 10; i2[1] = 10; i2[2] = 10;
    dimensions<3> dims(index_range<3>(i1, i2));
    block_index_space<3> bis(dims);
    mask<3> m;
    m[0] = true; m[1] = true; m[2] = true;
    bis.split(m, 2);
    bis.split(m, 5);

    block_tensor<3, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

    //  Fill in random input

    btod_random<3>().perform(bta);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<3, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
    tod_btconv<3>(bta).perform(ta);
    tod_add<3> refop(ta);
    refop.add_op(ta, permutation<3>().permute(0, 1), -1.0);
    refop.add_op(ta, permutation<3>().permute(0, 2), -1.0);
    refop.add_op(ta, permutation<3>().permute(1, 2), -1.0);
    refop.add_op(ta, permutation<3>().permute(0, 1).permute(0, 2), 1.0);
    refop.add_op(ta, permutation<3>().permute(0, 1).permute(1, 2), 1.0);
    refop.perform(true, tb_ref);

    //  Run the symmetrization operation

    btod_copy<3> op_copy(bta);
    btod_symmetrize3<3>(op_copy, 0, 1, 2, false).perform(btb);

    tod_btconv<3>(btb).perform(tb);

    //  Compare against the reference: symmetry and data

    symmetry<3, double> symb(bis), symb_ref(bis);
    {
        block_tensor_ctrl<3, double> ctrlb(btb);
        so_copy<3, double>(ctrlb.req_const_symmetry()).perform(symb);
    }
    scalar_transf<double> tr0, tr1(-1.);
    symb_ref.insert(se_perm<3, double>(
        permutation<3>().permute(0, 1), tr1));
    symb_ref.insert(se_perm<3, double>(
        permutation<3>().permute(0, 2), tr1));

    compare_ref<3>::compare(testname, symb, symb_ref);

    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


/** \test Symmetrization of a 3-index block %tensor with S(+)2*C1
        over three indexes
 **/
void btod_symmetrize3_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize3_test::test_3()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<3> i1, i2;
    i2[0] = 10; i2[1] = 10; i2[2] = 10;
    dimensions<3> dims(index_range<3>(i1, i2));
    block_index_space<3> bis(dims);
    mask<3> m;
    m[0] = true; m[1] = true; m[2] = true;
    bis.split(m, 2);
    bis.split(m, 5);

    block_tensor<3, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);
    scalar_transf<double> tr0, tr1(-1.);

    //  Set up initial symmetry and fill in random input

    {
        block_tensor_ctrl<3, double> ctrla(bta);
        ctrla.req_symmetry().insert(se_perm<3, double>(
            permutation<3>().permute(1, 2), tr0));
    }
    btod_random<3>().perform(bta);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<3, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
    tod_btconv<3>(bta).perform(ta);
    tod_add<3> refop(ta);
    refop.add_op(ta, permutation<3>().permute(0, 1), 1.0);
    refop.add_op(ta, permutation<3>().permute(0, 2), 1.0);
    refop.add_op(ta, permutation<3>().permute(1, 2), 1.0);
    refop.add_op(ta, permutation<3>().permute(0, 1).permute(0, 2), 1.0);
    refop.add_op(ta, permutation<3>().permute(0, 1).permute(1, 2), 1.0);
    refop.perform(true, tb_ref);

    //  Run the symmetrization operation

    btod_copy<3> op_copy(bta);
    btod_symmetrize3<3>(op_copy, 0, 1, 2, true).perform(btb);

    tod_btconv<3>(btb).perform(tb);

    //  Compare against the reference: symmetry and data

    symmetry<3, double> symb(bis), symb_ref(bis);
    {
        block_tensor_ctrl<3, double> ctrlb(btb);
        so_copy<3, double>(ctrlb.req_const_symmetry()).perform(symb);
    }
    symb_ref.insert(se_perm<3, double>(
        permutation<3>().permute(0, 1), tr0));
    symb_ref.insert(se_perm<3, double>(
        permutation<3>().permute(0, 1).permute(1, 2), tr0));

    compare_ref<3>::compare(testname, symb, symb_ref);

    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


/** \test Symmetrization of a 4-index block %tensor with S(+)2*C1
        over three indexes
 **/
void btod_symmetrize3_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize3_test::test_4()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<4> i1, i2;
    i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m;
    m[0] = true; m[1] = true; m[2] = true; m[3] = true;
    bis.split(m, 2);
    bis.split(m, 5);

    block_tensor<4, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

    //  Set up initial symmetry and fill in random input

    {
        block_tensor_ctrl<4, double> ctrla(bta);
        scalar_transf<double> tr0, tr1(-1.);
        ctrla.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(2, 3), tr0));
    }
    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<4, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
    tod_btconv<4>(bta).perform(ta);
    tod_add<4> refop(ta);
    refop.add_op(ta, permutation<4>().permute(0, 1), 1.0);
    refop.add_op(ta, permutation<4>().permute(0, 2), 1.0);
    refop.add_op(ta, permutation<4>().permute(1, 2), 1.0);
    refop.add_op(ta, permutation<4>().permute(0, 1).permute(0, 2), 1.0);
    refop.add_op(ta, permutation<4>().permute(0, 1).permute(1, 2), 1.0);
    refop.perform(true, tb_ref);

    //  Run the symmetrization operation

    btod_copy<4> op_copy(bta);
    btod_symmetrize3<4>(op_copy, 0, 1, 2, true).perform(btb);
    tod_btconv<4>(btb).perform(tb);

    //  Compare against the reference: symmetry and data

    symmetry<4, double> symb(bis), symb_ref(bis);
    {
        block_tensor_ctrl<4, double> ctrlb(btb);
        so_copy<4, double>(ctrlb.req_const_symmetry()).perform(symb);
    }
    scalar_transf<double> tr0, tr1(-1.);
    symb_ref.insert(se_perm<4, double>(
        permutation<4>().permute(0, 1), tr0));
    symb_ref.insert(se_perm<4, double>(
        permutation<4>().permute(0, 1).permute(1, 2), tr0));

    compare_ref<4>::compare(testname, symb, symb_ref);

    compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


/** \test Symmetrization of a 3-index block %tensor with partitions
        over three indexes
 **/
void btod_symmetrize3_test::test_5() throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize3_test::test_5()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<3> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9;
    dimensions<3> dims(index_range<3>(i1, i2));
    block_index_space<3> bis(dims);
    mask<3> m;
    m[0] = true; m[1] = true; m[2] = true;
    bis.split(m, 2);
    bis.split(m, 5);
    bis.split(m, 7);

    block_tensor<3, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

    //  Set up initial symmetry and fill in random input

    {
        libtensor::index<3> i000, i001, i010, i011, i100, i101, i110, i111;
        i110[0] = 1; i110[1] = 1; i001[2] = 1;
        i101[0] = 1; i010[1] = 1; i101[2] = 1;
        i100[0] = 1; i011[1] = 1; i011[2] = 1;
        i111[0] = 1; i111[1] = 1; i111[2] = 1;
        block_tensor_ctrl<3, double> ctrla(bta);
        se_part<3, double> p(bis, m, 2);
        scalar_transf<double> tr0, tr1(-1.);
        p.add_map(i000, i111, tr0);
        p.add_map(i001, i110, tr0);
        p.add_map(i010, i101, tr0);
        p.add_map(i011, i100, tr0);
        ctrla.req_symmetry().insert(p);
    }
    btod_random<3>().perform(bta);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<3, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
    tod_btconv<3>(bta).perform(ta);
    tod_add<3> refop(ta);
    refop.add_op(ta, permutation<3>().permute(0, 1), 1.0);
    refop.add_op(ta, permutation<3>().permute(0, 2), 1.0);
    refop.add_op(ta, permutation<3>().permute(1, 2), 1.0);
    refop.add_op(ta, permutation<3>().permute(0, 1).permute(0, 2), 1.0);
    refop.add_op(ta, permutation<3>().permute(0, 1).permute(1, 2), 1.0);
    refop.perform(true, tb_ref);

    symmetry<3, double> symb(bis), symb_ref(bis);
    scalar_transf<double> tr0, tr1(-1.);
    symb_ref.insert(se_perm<3, double>(
        permutation<3>().permute(0, 1), tr0));
    symb_ref.insert(se_perm<3, double>(
        permutation<3>().permute(0, 2), tr0));
    {
        libtensor::index<3> i000, i001, i010, i011, i100, i101, i110, i111;
        i110[0] = 1; i110[1] = 1; i001[2] = 1;
        i101[0] = 1; i010[1] = 1; i101[2] = 1;
        i100[0] = 1; i011[1] = 1; i011[2] = 1;
        i111[0] = 1; i111[1] = 1; i111[2] = 1;
        block_tensor_ctrl<3, double> ctrla(bta);
        se_part<3, double> p(bis, m, 2);
        p.add_map(i000, i111, tr0);
        p.add_map(i001, i110, tr0);
        p.add_map(i010, i101, tr0);
        p.add_map(i011, i100, tr0);
        symb_ref.insert(p);
    }

    //  Run the symmetrization operation

    btod_copy<3> op_copy(bta);
    btod_symmetrize3<3> op_sym(op_copy, 0, 1, 2, true);

    compare_ref<3>::compare(testname, op_sym.get_symmetry(), symb_ref);

    op_sym.perform(btb);
    tod_btconv<3>(btb).perform(tb);

    //  Compare against the reference: symmetry and data

    {
        block_tensor_ctrl<3, double> ctrlb(btb);
        so_copy<3, double>(ctrlb.req_const_symmetry()).perform(symb);
    }

    compare_ref<3>::compare(testname, symb, symb_ref);
    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


/** \test Double anti-symmetrization of a 6-index block %tensor with
        S(-)2*C1*C1*S(-)2 over three indexes
 **/
void btod_symmetrize3_test::test_6() throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize3_test::test_6()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<6> i1, i2;
    i2[0] = 5; i2[1] = 5; i2[2] = 5; i2[3] = 7; i2[4] = 7; i2[5] = 7;
    dimensions<6> dims(index_range<6>(i1, i2));
    block_index_space<6> bis(dims);

    block_tensor<6, double, allocator_t> bta(bis), btb(bis);
    scalar_transf<double> tr0, tr1(-1.);

    //  Set up initial symmetry and fill in random input

    {
        block_tensor_ctrl<6, double> ctrla(bta);
        ctrla.req_symmetry().insert(se_perm<6, double>(
            permutation<6>().permute(0, 1), tr1));
        ctrla.req_symmetry().insert(se_perm<6, double>(
            permutation<6>().permute(4, 5), tr1));
    }

    btod_random<6>().perform(bta);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<6, double, allocator_t> ta(dims), ta1(dims), tb(dims),
        tb_ref(dims);
    tod_btconv<6>(bta).perform(ta);
    tod_add<6> refop1(ta);
    refop1.add_op(ta, permutation<6>().permute(0, 1), -1.0);
    refop1.add_op(ta, permutation<6>().permute(0, 2), -1.0);
    refop1.add_op(ta, permutation<6>().permute(1, 2), -1.0);
    refop1.add_op(ta, permutation<6>().permute(0, 1).permute(0, 2), 1.0);
    refop1.add_op(ta, permutation<6>().permute(0, 1).permute(1, 2), 1.0);
    refop1.perform(true, ta1);
    tod_add<6> refop2(ta1);
    refop2.add_op(ta1, permutation<6>().permute(3, 4), -1.0);
    refop2.add_op(ta1, permutation<6>().permute(3, 5), -1.0);
    refop2.add_op(ta1, permutation<6>().permute(4, 5), -1.0);
    refop2.add_op(ta1, permutation<6>().permute(3, 4).permute(3, 5), 1.0);
    refop2.add_op(ta1, permutation<6>().permute(3, 4).permute(4, 5), 1.0);
    refop2.perform(true, tb_ref);

    //  Run the symmetrization operation

    btod_copy<6> op_copy(bta);
    btod_symmetrize3<6> op_sym3(op_copy, 0, 1, 2, false);
    btod_symmetrize3<6>(op_sym3, 3, 4, 5, false).perform(btb);

    tod_btconv<6>(btb).perform(tb);

    //  Compare against the reference: symmetry and data

    symmetry<6, double> symb(bis), symb_ref(bis);
    {
        block_tensor_ctrl<6, double> ctrlb(btb);
        so_copy<6, double>(ctrlb.req_const_symmetry()).perform(symb);
    }
    symb_ref.insert(se_perm<6, double>(
        permutation<6>().permute(0, 1), tr1));
    symb_ref.insert(se_perm<6, double>(
        permutation<6>().permute(0, 1).permute(1, 2), tr0));
    symb_ref.insert(se_perm<6, double>(
        permutation<6>().permute(3, 4), tr1));
    symb_ref.insert(se_perm<6, double>(
        permutation<6>().permute(3, 4).permute(4, 5), tr0));

    compare_ref<6>::compare(testname, symb, symb_ref);

    compare_ref<6>::compare(testname, tb, tb_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


/** \test Double anti-symmetrization of a 6-index block %tensor with
        S(-)2*C1*C1*S(-)2 over three indexes (additive)
 **/
void btod_symmetrize3_test::test_7() throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize3_test::test_7()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<6> i1, i2;
    i2[0] = 5; i2[1] = 5; i2[2] = 5; i2[3] = 7; i2[4] = 7; i2[5] = 7;
    dimensions<6> dims(index_range<6>(i1, i2));
    block_index_space<6> bis(dims);

    block_tensor<6, double, allocator_t> bta(bis), btb(bis);
    scalar_transf<double> tr0, tr1(-1.);

    //  Set up initial symmetry and fill in random input

    {
        block_tensor_ctrl<6, double> ctrla(bta), ctrlb(btb);
        ctrla.req_symmetry().insert(se_perm<6, double>(
            permutation<6>().permute(0, 1), tr1));
        ctrla.req_symmetry().insert(se_perm<6, double>(
            permutation<6>().permute(4, 5), tr1));
        ctrlb.req_symmetry().insert(se_perm<6, double>(
            permutation<6>().permute(0, 1), tr1));
        ctrlb.req_symmetry().insert(se_perm<6, double>(
            permutation<6>().permute(1, 2), tr1));
        ctrlb.req_symmetry().insert(se_perm<6, double>(
            permutation<6>().permute(3, 4), tr1));
        ctrlb.req_symmetry().insert(se_perm<6, double>(
            permutation<6>().permute(4, 5), tr1));
    }

    btod_random<6>().perform(bta);
    btod_random<6>().perform(btb);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<6, double, allocator_t> ta(dims), ta1(dims), tb(dims),
        tb_ref(dims);
    tod_btconv<6>(bta).perform(ta);
    tod_btconv<6>(btb).perform(tb_ref);
    tod_add<6> refop1(ta);
    refop1.add_op(ta, permutation<6>().permute(0, 1), -1.0);
    refop1.add_op(ta, permutation<6>().permute(0, 2), -1.0);
    refop1.add_op(ta, permutation<6>().permute(1, 2), -1.0);
    refop1.add_op(ta, permutation<6>().permute(0, 1).permute(0, 2), 1.0);
    refop1.add_op(ta, permutation<6>().permute(0, 1).permute(1, 2), 1.0);
    refop1.perform(true, ta1);
    tod_add<6> refop2(ta1, 2.0);
    refop2.add_op(ta1, permutation<6>().permute(3, 4), -2.0);
    refop2.add_op(ta1, permutation<6>().permute(3, 5), -2.0);
    refop2.add_op(ta1, permutation<6>().permute(4, 5), -2.0);
    refop2.add_op(ta1, permutation<6>().permute(3, 4).permute(3, 5), 2.0);
    refop2.add_op(ta1, permutation<6>().permute(3, 4).permute(4, 5), 2.0);
    refop2.perform(false, tb_ref);

    //  Run the symmetrization operation

    btod_copy<6> op_copy(bta);
    btod_symmetrize3<6> op_sym3(op_copy, 0, 1, 2, false);
    btod_symmetrize3<6>(op_sym3, 3, 4, 5, false).perform(btb, 2.0);

    tod_btconv<6>(btb).perform(tb);

    //  Compare against the reference: symmetry and data

    symmetry<6, double> symb(bis), symb_ref(bis);
    {
        block_tensor_ctrl<6, double> ctrlb(btb);
        so_copy<6, double>(ctrlb.req_const_symmetry()).perform(symb);
    }
    symb_ref.insert(se_perm<6, double>(
        permutation<6>().permute(0, 1), tr1));
    symb_ref.insert(se_perm<6, double>(
        permutation<6>().permute(0, 1).permute(1, 2), tr0));
    symb_ref.insert(se_perm<6, double>(
        permutation<6>().permute(3, 4), tr1));
    symb_ref.insert(se_perm<6, double>(
        permutation<6>().permute(3, 4).permute(4, 5), tr0));

    compare_ref<6>::compare(testname, symb, symb_ref);

    compare_ref<6>::compare(testname, tb, tb_ref, 2e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


/** \test Anti-symmetrization of a contraction that appears in CCSD(T)
 **/
void btod_symmetrize3_test::test_8a() throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize3_test::test_8a()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<4> i4a, i4b;
    i4b[0] = 4; i4b[1] = 4; i4b[2] = 6; i4b[3] = 6;
    dimensions<4> dims_oovv(index_range<4>(i4a, i4b));
    block_index_space<4> bis_oovv(dims_oovv);
    mask<4> m0011, m1100;
    m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
    bis_oovv.split(m1100, 2);
    bis_oovv.split(m1100, 3);
    bis_oovv.split(m0011, 3);
    bis_oovv.split(m0011, 4);
    i4b[0] = 4; i4b[1] = 6; i4b[2] = 6; i4b[3] = 6;
    dimensions<4> dims_ovvv(index_range<4>(i4a, i4b));
    block_index_space<4> bis_ovvv(dims_ovvv);
    mask<4> m0111, m1000;
    m1000[0] = true; m0111[1] = true; m0111[2] = true; m0111[3] = true;
    bis_ovvv.split(m1000, 2);
    bis_ovvv.split(m1000, 3);
    bis_ovvv.split(m0111, 3);
    bis_ovvv.split(m0111, 4);
    libtensor::index<6> i6a, i6b;
    i6b[0] = 4; i6b[1] = 4; i6b[2] = 4; i6b[3] = 6; i6b[4] = 6; i6b[5] = 6;
    dimensions<6> dims_ooovvv(index_range<6>(i6a, i6b));
    block_index_space<6> bis_ooovvv(dims_ooovvv);
    mask<6> m000111, m111000;
    m111000[0] = true; m111000[1] = true; m111000[2] = true;
    m000111[3] = true; m000111[4] = true; m000111[5] = true;
    bis_ooovvv.split(m111000, 2);
    bis_ooovvv.split(m111000, 3);
    bis_ooovvv.split(m000111, 3);
    bis_ooovvv.split(m000111, 4);

    scalar_transf<double> tr0(1.0), tr1(-1.0);

    block_tensor<4, double, allocator_t> bta(bis_oovv), btb(bis_ovvv);
    block_tensor<6, double, allocator_t> btc(bis_ooovvv);

    {
        block_tensor_ctrl<4, double> ca(bta);
        ca.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr1));
        ca.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(2, 3), tr1));
    }
    {
        block_tensor_ctrl<4, double> cb(btb);
        cb.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(2, 3), tr1));
    }

    //  Fill in random input

    btod_random<4>().perform(bta);
    btod_random<4>().perform(btb);
    bta.set_immutable();
    btb.set_immutable();

    //  Prepare reference data

    dense_tensor<4, double, allocator_t> ta(dims_oovv), tb(dims_ovvv);
    dense_tensor<6, double, allocator_t> tt1(dims_ooovvv), tt2(dims_ooovvv),
        tc(dims_ooovvv), tc_ref(dims_ooovvv);
    tod_btconv<4>(bta).perform(ta);
    tod_btconv<4>(btb).perform(tb);

    contraction2<3, 3, 1> contr(permutation<6>().permute(2, 3).permute(4, 5));
    contr.contract(3, 1);

    tod_contract2<3, 3, 1>(contr, ta, tb).perform(true, tt1);
    tod_add<6> refop1(tt1);
    refop1.add_op(tt1, permutation<6>().permute(0, 1), -1.0);
    refop1.add_op(tt1, permutation<6>().permute(0, 2), -1.0);
    refop1.add_op(tt1, permutation<6>().permute(1, 2), -1.0);
    refop1.add_op(tt1, permutation<6>().permute(0, 1).permute(0, 2), 1.0);
    refop1.add_op(tt1, permutation<6>().permute(0, 1).permute(1, 2), 1.0);
    refop1.perform(true, tt2);
    tod_add<6> refop2(tt2);
    refop2.add_op(tt2, permutation<6>().permute(3, 4), -1.0);
    refop2.add_op(tt2, permutation<6>().permute(3, 5), -1.0);
    refop2.add_op(tt2, permutation<6>().permute(4, 5), -1.0);
    refop2.add_op(tt2, permutation<6>().permute(3, 4).permute(3, 5), 1.0);
    refop2.add_op(tt2, permutation<6>().permute(3, 4).permute(4, 5), 1.0);
    refop2.perform(true, tc_ref);

    symmetry<6, double> symc(bis_ooovvv), symc_ref(bis_ooovvv);
    symc_ref.insert(se_perm<6, double>(permutation<6>().permute(0, 1), tr1));
    symc_ref.insert(se_perm<6, double>(permutation<6>().permute(0, 2), tr1));
    symc_ref.insert(se_perm<6, double>(permutation<6>().permute(3, 4), tr1));
    symc_ref.insert(se_perm<6, double>(permutation<6>().permute(3, 5), tr1));

    //  Run the symmetrization operation

    btod_contract2<3, 3, 1> op1(contr, bta, btb);
    btod_symmetrize3<6> sym1a(op1, 3, 4, 5, false);
    btod_symmetrize3<6> sym1b(sym1a, 0, 1, 2, false);
    sym1b.perform(btc);
    tod_btconv<6>(btc).perform(tc);

    //  Compare against the reference: symmetry and data

    {
        block_tensor_ctrl<6, double> cc(btc);
        so_copy<6, double>(cc.req_const_symmetry()).perform(symc);
    }

    compare_ref<6>::compare(testname, symc, symc_ref);
    compare_ref<6>::compare(testname, tc, tc_ref, 1e-14);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


/** \test Anti-symmetrization of a contraction that appears in CCSD(T),
        invoked blockwise
 **/
void btod_symmetrize3_test::test_8b() throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize3_test::test_8b()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<4> i4a, i4b;
    i4b[0] = 4; i4b[1] = 4; i4b[2] = 6; i4b[3] = 6;
    dimensions<4> dims_oovv(index_range<4>(i4a, i4b));
    block_index_space<4> bis_oovv(dims_oovv);
    mask<4> m0011, m1100;
    m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
    bis_oovv.split(m1100, 2);
    bis_oovv.split(m1100, 3);
    bis_oovv.split(m0011, 3);
    bis_oovv.split(m0011, 4);
    i4b[0] = 4; i4b[1] = 6; i4b[2] = 6; i4b[3] = 6;
    dimensions<4> dims_ovvv(index_range<4>(i4a, i4b));
    block_index_space<4> bis_ovvv(dims_ovvv);
    mask<4> m0111, m1000;
    m1000[0] = true; m0111[1] = true; m0111[2] = true; m0111[3] = true;
    bis_ovvv.split(m1000, 2);
    bis_ovvv.split(m1000, 3);
    bis_ovvv.split(m0111, 3);
    bis_ovvv.split(m0111, 4);
    libtensor::index<6> i6a, i6b;
    i6b[0] = 4; i6b[1] = 4; i6b[2] = 4; i6b[3] = 6; i6b[4] = 6; i6b[5] = 6;
    dimensions<6> dims_ooovvv(index_range<6>(i6a, i6b));
    block_index_space<6> bis_ooovvv(dims_ooovvv);
    mask<6> m000111, m111000;
    m111000[0] = true; m111000[1] = true; m111000[2] = true;
    m000111[3] = true; m000111[4] = true; m000111[5] = true;
    bis_ooovvv.split(m111000, 2);
    bis_ooovvv.split(m111000, 3);
    bis_ooovvv.split(m000111, 3);
    bis_ooovvv.split(m000111, 4);

    scalar_transf<double> tr0(1.0), tr1(-1.0);

    block_tensor<4, double, allocator_t> bta(bis_oovv), btb(bis_ovvv);
    block_tensor<6, double, allocator_t> btc(bis_ooovvv);

    {
        block_tensor_ctrl<4, double> ca(bta);
        ca.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr1));
        ca.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(2, 3), tr1));
    }
    {
        block_tensor_ctrl<4, double> cb(btb);
        cb.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(2, 3), tr1));
    }

    //  Fill in random input

    btod_random<4>().perform(bta);
    btod_random<4>().perform(btb);
    bta.set_immutable();
    btb.set_immutable();

    //  Prepare reference data

    dense_tensor<4, double, allocator_t> ta(dims_oovv), tb(dims_ovvv);
    dense_tensor<6, double, allocator_t> tt1(dims_ooovvv), tt2(dims_ooovvv),
        tc(dims_ooovvv), tc_ref(dims_ooovvv);
    tod_btconv<4>(bta).perform(ta);
    tod_btconv<4>(btb).perform(tb);

    contraction2<3, 3, 1> contr(permutation<6>().permute(2, 3).permute(4, 5));
    contr.contract(3, 1);

    tod_contract2<3, 3, 1>(contr, ta, tb).perform(true, tt1);
    tod_add<6> refop1(tt1);
    refop1.add_op(tt1, permutation<6>().permute(0, 1), -1.0);
    refop1.add_op(tt1, permutation<6>().permute(0, 2), -1.0);
    refop1.add_op(tt1, permutation<6>().permute(1, 2), -1.0);
    refop1.add_op(tt1, permutation<6>().permute(0, 1).permute(0, 2), 1.0);
    refop1.add_op(tt1, permutation<6>().permute(0, 1).permute(1, 2), 1.0);
    refop1.perform(true, tt2);
    tod_add<6> refop2(tt2);
    refop2.add_op(tt2, permutation<6>().permute(3, 4), -1.0);
    refop2.add_op(tt2, permutation<6>().permute(3, 5), -1.0);
    refop2.add_op(tt2, permutation<6>().permute(4, 5), -1.0);
    refop2.add_op(tt2, permutation<6>().permute(3, 4).permute(3, 5), 1.0);
    refop2.add_op(tt2, permutation<6>().permute(3, 4).permute(4, 5), 1.0);
    refop2.perform(true, tc_ref);

    symmetry<6, double> symc(bis_ooovvv), symc_ref(bis_ooovvv);
    symc_ref.insert(se_perm<6, double>(permutation<6>().permute(0, 1), tr1));
    symc_ref.insert(se_perm<6, double>(permutation<6>().permute(0, 2), tr1));
    symc_ref.insert(se_perm<6, double>(permutation<6>().permute(3, 4), tr1));
    symc_ref.insert(se_perm<6, double>(permutation<6>().permute(3, 5), tr1));

    //  Run the symmetrization operation

    btod_contract2<3, 3, 1> op1(contr, bta, btb);
    btod_symmetrize3<6> sym1a(op1, 3, 4, 5, false);
    btod_symmetrize3<6> sym1b(sym1a, 0, 1, 2, false);

    dimensions<6> bidims = bis_ooovvv.get_block_index_dims();
    block_tensor_ctrl<6, double> cc(btc);
    const assignment_schedule<6, double> &asch = sym1b.get_schedule();
    for(assignment_schedule<6, double>::iterator i = asch.begin();
        i != asch.end(); ++i) {
        libtensor::index<6> idx;
        abs_index<6>::get_index(asch.get_abs_index(i), bidims, idx);
        dense_tensor_wr_i<6, double> &blk = cc.req_block(idx);
        sym1b.compute_block(idx, blk);
        cc.ret_block(idx);
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


/** \test Symmetrization of a 4-index block %tensor with S(+)2*S(+)2
        over three indexes
 **/
void btod_symmetrize3_test::test_9() throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize3_test::test_9()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<4> i1, i2;
    i2[0] = 10; i2[1] = 10; i2[2] = 10; i2[3] = 10;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m;
    m[0] = true; m[1] = true; m[2] = true; m[3] = true;
    bis.split(m, 2);
    bis.split(m, 5);

    block_tensor<4, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

    //  Set up initial symmetry and fill in random input

    {
        block_tensor_ctrl<4, double> ctrla(bta);
        scalar_transf<double> tr0, tr1(-1.);
        ctrla.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 1), tr0));
        ctrla.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(2, 3), tr0));
    }
    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<4, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
    tod_btconv<4>(bta).perform(ta);
    tod_add<4> refop(ta);
    refop.add_op(ta, permutation<4>().permute(0, 1), 1.0);
    refop.add_op(ta, permutation<4>().permute(0, 2), 1.0);
    refop.add_op(ta, permutation<4>().permute(1, 2), 1.0);
    refop.add_op(ta, permutation<4>().permute(0, 1).permute(0, 2), 1.0);
    refop.add_op(ta, permutation<4>().permute(0, 1).permute(1, 2), 1.0);
    refop.perform(true, tb_ref);

    //  Run the symmetrization operation

    btod_copy<4> op_copy(bta);
    btod_symmetrize3<4>(op_copy, 0, 1, 2, true).perform(btb);
    tod_btconv<4>(btb).perform(tb);

    //  Compare against the reference: symmetry and data

    symmetry<4, double> symb(bis), symb_ref(bis);
    {
        block_tensor_ctrl<4, double> ctrlb(btb);
        so_copy<4, double>(ctrlb.req_const_symmetry()).perform(symb);
    }
    scalar_transf<double> tr0, tr1(-1.);
    symb_ref.insert(se_perm<4, double>(
        permutation<4>().permute(0, 1), tr0));
    symb_ref.insert(se_perm<4, double>(
        permutation<4>().permute(0, 1).permute(1, 2), tr0));

    compare_ref<4>::compare(testname, symb, symb_ref);

    compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


/** \test Symmetrization of a non-symmetric 3-index block %tensor
        over three indexes specified using permutations
 **/
void btod_symmetrize3_test::test_10() throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize3_test::test_10()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<3> i1, i2;
    i2[0] = 10; i2[1] = 10; i2[2] = 10;
    dimensions<3> dims(index_range<3>(i1, i2));
    block_index_space<3> bis(dims);
    mask<3> m;
    m[0] = true; m[1] = true; m[2] = true;
    bis.split(m, 2);
    bis.split(m, 5);

    block_tensor<3, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

    //  Fill in random input

    btod_random<3>().perform(bta);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<3, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
    tod_btconv<3>(bta).perform(ta);
    tod_add<3> refop(ta);
    refop.add_op(ta, permutation<3>().permute(0, 1), 1.0);
    refop.add_op(ta, permutation<3>().permute(0, 2), 1.0);
    refop.add_op(ta, permutation<3>().permute(1, 2), 1.0);
    refop.add_op(ta, permutation<3>().permute(0, 1).permute(0, 2), 1.0);
    refop.add_op(ta, permutation<3>().permute(0, 1).permute(1, 2), 1.0);
    refop.perform(true, tb_ref);

    symmetry<3, double> symb(bis), symb_ref(bis);
    scalar_transf<double> tr0, tr1(-1.);
    symb_ref.insert(se_perm<3, double>(
        permutation<3>().permute(0, 1), tr0));
    symb_ref.insert(se_perm<3, double>(
        permutation<3>().permute(0, 2), tr0));

    //  Run the symmetrization operation

    btod_copy<3> op_copy(bta);
    btod_symmetrize3<3> op_sym(op_copy, permutation<3>().permute(0, 1),
        permutation<3>().permute(0, 2), true);

    compare_ref<3>::compare(testname, op_sym.get_symmetry(), symb_ref);

    op_sym.perform(btb);
    tod_btconv<3>(btb).perform(tb);

    //  Compare against the reference: symmetry and data

    {
        block_tensor_ctrl<3, double> ctrlb(btb);
        so_copy<3, double>(ctrlb.req_const_symmetry()).perform(symb);
    }

    compare_ref<3>::compare(testname, symb, symb_ref);
    compare_ref<3>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


} // namespace libtensor
