#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/btod_add.h>
#include <libtensor/block_tensor/btod_contract2.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/block_tensor/btod_symmetrize2.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/dense_tensor/tod_btconv.h>
#include "btod_symmetrize2_test.h"
#include "../compare_ref.h"

namespace libtensor {


void btod_symmetrize2_test::perform() throw(libtest::test_exception) {

    allocator<double>::init();

    try {

        test_1();
        test_2();
        test_3();
        test_4();
        test_5(false);
        test_5(true);
        test_6a(false, false, false, false);
        test_6a(false, false, false, true);
        test_6a(false, false, true, false);
        test_6a(false, false, true, true);
        test_6a(false, true, false, false);
        test_6a(false, true, false, true);
        test_6a(false, true, true, false);
        test_6a(false, true, true, true);
        test_6a(true, false, false, false);
        test_6a(true, false, false, true);
        test_6a(true, false, true, false);
        test_6a(true, false, true, true);
        test_6a(true, true, false, false);
        test_6a(true, true, false, true);
        test_6a(true, true, true, false);
        test_6a(true, true, true, true);
        test_6b(false, false, false);
        test_6b(false, false, true);
        test_6b(false, true, false);
        test_6b(false, true, true);
        test_6b(true, false, false);
        test_6b(true, false, true);
        test_6b(true, true, false);
        test_6b(true, true, true);
        test_7();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}

/** \test Symmetrization of a non-symmetric 2-index block %tensor
 **/
void btod_symmetrize2_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize2_test::test_1()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 10; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m;
    m[0] = true; m[1] = true;
    bis.split(m, 2);
    bis.split(m, 5);

    block_tensor<2, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

    //  Fill in random input

    btod_random<2>().perform(bta);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
    tod_btconv<2>(bta).perform(ta);
    tod_add<2> refop(ta);
    refop.add_op(ta, permutation<2>().permute(0, 1), 1.0);
    refop.perform(true, tb_ref);

    //  Run the symmetrization operation

    btod_copy<2> op_copy(bta);
    btod_symmetrize2<2>(op_copy, 0, 1, true).perform(btb);

    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference: symmetry and data

    symmetry<2, double> symb(bis), symb_ref(bis);
    {
        block_tensor_ctrl<2, double> ctrlb(btb);
        so_copy<2, double>(ctrlb.req_const_symmetry()).perform(symb);
    }
    scalar_transf<double> tr0, tr1(-1.);
    symb_ref.insert(se_perm<2, double>(
        permutation<2>().permute(0, 1), tr0));

    compare_ref<2>::compare(testname, symb, symb_ref);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Anti-symmetrization of a non-symmetric 2-index block %tensor
 **/
void btod_symmetrize2_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize2_test::test_2()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 10; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m;
    m[0] = true; m[1] = true;
    bis.split(m, 2);
    bis.split(m, 5);

    block_tensor<2, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

    //  Fill in random input

    btod_random<2>().perform(bta);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
    tod_btconv<2>(bta).perform(ta);
    tod_add<2> refop(ta);
    refop.add_op(ta, permutation<2>().permute(0, 1), -1.0);
    refop.perform(true, tb_ref);

    //  Run the symmetrization operation

    btod_copy<2> op_copy(bta);
    btod_symmetrize2<2>(op_copy, 0, 1, false).perform(btb);

    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference: symmetry and data

    symmetry<2, double> symb(bis), symb_ref(bis);
    {
        block_tensor_ctrl<2, double> ctrlb(btb);
        so_copy<2, double>(ctrlb.req_const_symmetry()).perform(symb);
    }
    scalar_transf<double> tr0, tr1(-1.);
    symb_ref.insert(se_perm<2, double>(
        permutation<2>().permute(0, 1), tr1));

    compare_ref<2>::compare(testname, symb, symb_ref);

    compare_ref<2>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Anti-symmetrization of S(-)2*C1*C1 to S(-)2*S(-)2
        in a 4-index block %tensor
 **/
void btod_symmetrize2_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize2_test::test_3()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<4> i1, i2;
    i2[0] = 20; i2[1] = 10; i2[2] = 20; i2[3] = 10;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m1, m2;
    m1[0] = true; m2[1] = true; m1[2] = true; m2[3] = true;
    bis.split(m1, 5);
    bis.split(m1, 10);
    bis.split(m2, 2);
    bis.split(m2, 5);

    block_tensor<4, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

    //  Set up initial symmetry and fill in random input

    {
        scalar_transf<double> tr0, tr1(-1.);
        block_tensor_ctrl<4, double> ctrla(bta);
        ctrla.req_symmetry().insert(se_perm<4, double>(
            permutation<4>().permute(0, 2), tr1));
    }
    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<4, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
    tod_btconv<4>(bta).perform(ta);
    tod_add<4> refop(ta);
    refop.add_op(ta, permutation<4>().permute(1, 3), -1.0);
    refop.perform(true, tb_ref);

    //  Run the symmetrization operation

    btod_copy<4> op_copy(bta);
    btod_symmetrize2<4>(op_copy, 1, 3, false).perform(btb);

    tod_btconv<4>(btb).perform(tb);

    //  Compare against the reference: symmetry and data

    symmetry<4, double> symb(bis), symb_ref(bis);
    {
        block_tensor_ctrl<4, double> ctrlb(btb);
        so_copy<4, double>(ctrlb.req_const_symmetry()).perform(symb);
    }
    scalar_transf<double> tr0, tr1(-1.);
    symb_ref.insert(se_perm<4, double>(
        permutation<4>().permute(0, 2), tr1));
    symb_ref.insert(se_perm<4, double>(
        permutation<4>().permute(1, 3), tr1));

    compare_ref<4>::compare(testname, symb, symb_ref);

    compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Symmetrization of S2*S2 to S2*C1*C1 in a 4-index block %tensor
 **/
void btod_symmetrize2_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize2_test::test_4()";

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
    scalar_transf<double> tr0;

    {
        block_tensor_ctrl<4, double> ctrla(bta);
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
    refop.add_op(ta, permutation<4>().permute(0, 2), 1.0);
    refop.perform(true, tb_ref);

    //  Run the symmetrization operation

    btod_copy<4> op_copy(bta);
    btod_symmetrize2<4>(op_copy, 0, 2, true).perform(btb);

    tod_btconv<4>(btb).perform(tb);

    //  Compare against the reference: symmetry and data

    symmetry<4, double> symb(bis), symb_ref(bis);
    {
        block_tensor_ctrl<4, double> ctrlb(btb);
        so_copy<4, double>(ctrlb.req_const_symmetry()).perform(symb);
    }
    symb_ref.insert(se_perm<4, double>(
        permutation<4>().permute(0, 2), tr0));

    compare_ref<4>::compare(testname, symb, symb_ref);

    compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Symmetrization of two pairs of indexes in a non-symmetric
        4-index block %tensor
 **/
void btod_symmetrize2_test::test_5(bool symm) throw(libtest::test_exception) {

    static const char *testname = "btod_symmetrize2_test::test_5(bool)";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<4> i1, i2;
    i2[0] = 20; i2[1] = 20; i2[2] = 20; i2[3] = 20;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m;
    m[0] = true; m[1] = true; m[2] = true; m[3] = true;
    bis.split(m, 5);
    bis.split(m, 10);
    bis.split(m, 15);

    block_tensor<4, double, allocator_t> bta(bis), btb(bis), btb_ref(bis);

    //  Fill in random input

    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<4, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
    tod_btconv<4>(bta).perform(ta);
    tod_add<4> refop(ta);
    refop.add_op(ta, permutation<4>().permute(0, 2).permute(1, 3),
            (symm ? 1.0 : -1.0));
    refop.perform(true, tb_ref);

    //  Run the symmetrization operation

    btod_copy<4> op_copy(bta);
    btod_symmetrize2<4>(op_copy, permutation<4>().permute(0, 2).
        permute(1, 3), symm).perform(btb);

    tod_btconv<4>(btb).perform(tb);

    //  Compare against the reference: symmetry and data

    symmetry<4, double> symb(bis), symb_ref(bis);
    {
        block_tensor_ctrl<4, double> ctrlb(btb);
        so_copy<4, double>(ctrlb.req_const_symmetry()).perform(symb);
    }
    scalar_transf<double> tr0, tr1(-1.);
    symb_ref.insert(se_perm<4, double>(
        permutation<4>().permute(0, 2).permute(1, 3), symm ? tr0 : tr1));

    compare_ref<4>::compare(testname, symb, symb_ref);

    compare_ref<4>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test (Anti-)Symmetrization of two indexes in a non-symmetric
        2-dim block %tensor with se_label, se_part
 **/
void btod_symmetrize2_test::test_6a(bool symm, bool label,
        bool part, bool doadd) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "btod_symmetrize2_test::test_6a(" << symm << ", "
            << label << ", " << part << ", " << doadd << ")";
    std::string tns = tnss.str();

    typedef allocator<double> allocator_t;

    if (label) {
        std::vector<std::string> irnames(2);
        irnames[0] = "g"; irnames[1] = "u";
        point_group_table pg(tns, irnames, irnames[0]);
        pg.add_product(1, 1, 0);
        product_table_container::get_instance().add(pg);
    }

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 11; i2[1] = 11;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m;
    m[0] = true; m[1] = true;
    bis.split(m, 2);
    bis.split(m, 6);
    bis.split(m, 8);

    permutation<2> p;
    p.permute(0, 1);

    block_tensor<2, double, allocator_t> bta(bis), btb(bis);
    symmetry<2, double> sym_ref(bis);

    // setup symmetry
    {
    block_tensor_ctrl<2, double> ca(bta), cb(btb);

    scalar_transf<double> tr(symm ? 1.0 : -1.0);
    se_perm<2, double> se10(p, tr);
    cb.req_symmetry().insert(se10);
    sym_ref.insert(se10);

    if (label) {
        se_label<2, double> sl(bis.get_block_index_dims(), tns);
        block_labeling<2> &bl = sl.get_labeling();
        bl.assign(m, 0, 0);
        bl.assign(m, 1, 1);
        bl.assign(m, 2, 0);
        bl.assign(m, 3, 1);
        sl.set_rule(1);
        ca.req_symmetry().insert(sl);
        cb.req_symmetry().insert(sl);
        sym_ref.insert(sl);
    }

    if (part) {
        se_part<2, double> sp(bis, m, 2);
        libtensor::index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;
        sp.add_map(i00, i01);
        sp.add_map(i01, i10);
        sp.add_map(i10, i11);
        ca.req_symmetry().insert(sp);
        cb.req_symmetry().insert(sp);
        sym_ref.insert(sp);
    }
    }
    //  Fill in random input

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<2, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
    tod_btconv<2>(bta).perform(ta);
    double k = (doadd ? 0.25 : 1.0);
    tod_add<2> refop(ta, k);
    refop.add_op(ta, p, (symm ? 1.0 : -1.0) * k);
    if (doadd) {
        tod_btconv<2>(btb).perform(tb_ref);
        refop.perform(false, tb_ref);
    }
    else {
        refop.perform(true, tb_ref);
    }

    //  Run the symmetrization operation

    btod_copy<2> op_copy(bta);
    if (doadd) btod_symmetrize2<2>(op_copy, p, symm).perform(btb, 0.25);
    else btod_symmetrize2<2>(op_copy, p, symm).perform(btb);

    tod_btconv<2>(btb).perform(tb);

    //  Compare against the reference: symmetry and data
    block_tensor_ctrl<2, double> ctrlb(btb);

    compare_ref<2>::compare(tns.c_str(), ctrlb.req_const_symmetry(), sym_ref);

    compare_ref<2>::compare(tns.c_str(), tb, tb_ref, 1e-15);

    } catch(exception &e) {
        if (label) product_table_container::get_instance().erase(tns);

        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    } catch(...) {
        if (label) product_table_container::get_instance().erase(tns);

        throw;
    }
    if (label) product_table_container::get_instance().erase(tns);

}

/** \test Double (anti-)symmetrization of two indexes in a non-symmetric
        4-dim block %tensor with se_label, se_part
 **/
void btod_symmetrize2_test::test_6b(bool symm, bool label,
        bool part) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "btod_symmetrize2_test::test_6b(" << symm << ", "
            << label << ", " << part << ", " << ")";
    std::string tns = tnss.str();

    typedef allocator<double> allocator_t;

    if (label) {
        std::vector<std::string> irnames(2);
        irnames[0] = "g"; irnames[1] = "u";
        point_group_table pg(tns, irnames, irnames[0]);
        pg.add_product(1, 1, 0);

        product_table_container::get_instance().add(pg);
    }

    try {

    libtensor::index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m, m1, m2;
    m1[0] = true; m1[1] = true;
    m2[2] = true; m2[3] = true;
    m[0] = true; m[1] = true; m[2] = true; m[3] = true;
    bis.split(m1, 4); bis.split(m1, 5); bis.split(m1, 9);
    bis.split(m2, 1); bis.split(m2, 6); bis.split(m2, 7);

    permutation<4> p1, p2;
    p1.permute(0, 1);
    p2.permute(2, 3);

    block_tensor<4, double, allocator_t> bta(bis), btb(bis);
    symmetry<4, double> sym_ref(bis);

    // setup symmetry
    {
    block_tensor_ctrl<4, double> ca(bta);

    scalar_transf<double> tr(symm ? 1.0 : -1.0);
    se_perm<4, double> se1(p1, tr);
    se_perm<4, double> se2(p2, tr);
    sym_ref.insert(se1);
    sym_ref.insert(se2);

    if (label) {
        se_label<4, double> sl(bis.get_block_index_dims(), tns);
        block_labeling<4> &bl = sl.get_labeling();
        bl.assign(m, 0, 0);
        bl.assign(m, 1, 1);
        bl.assign(m, 2, 0);
        bl.assign(m, 3, 1);
        product_table_i::label_set_t ls;
        ls.insert(0); ls.insert(1);
        sl.set_rule(ls);
        ca.req_symmetry().insert(sl);
        sym_ref.insert(sl);
    }

    if (part) {
        se_part<4, double> sp(bis, m, 2);
        libtensor::index<4> i0000, i0001, i0010, i0011, i0100, i0101, i0110, i0111,
            i1000, i1001, i1010, i1011, i1100, i1101, i1110, i1111;
        i1110[0] = 1; i1110[1] = 1; i1110[2] = 1; i0001[3] = 1;
        i1101[0] = 1; i1101[1] = 1; i0010[2] = 1; i1101[3] = 1;
        i1100[0] = 1; i1100[1] = 1; i0011[2] = 1; i0011[3] = 1;
        i1011[0] = 1; i0100[1] = 1; i1011[2] = 1; i1011[3] = 1;
        i1010[0] = 1; i0101[1] = 1; i1010[2] = 1; i0101[3] = 1;
        i1001[0] = 1; i0110[1] = 1; i0110[2] = 1; i1001[3] = 1;
        i1000[0] = 1; i0111[1] = 1; i0111[2] = 1; i0111[3] = 1;
        i1111[0] = 1; i1111[1] = 1; i1111[2] = 1; i1111[3] = 1;
        sp.add_map(i0000, i0001);
        sp.add_map(i0001, i0010);
        sp.add_map(i0010, i0011);
        sp.add_map(i0011, i0100);
        sp.add_map(i0100, i0101);
        sp.add_map(i0101, i0110);
        sp.add_map(i0110, i0111);
        sp.add_map(i0111, i1000);
        sp.add_map(i1000, i1001);
        sp.add_map(i1001, i1010);
        sp.add_map(i1010, i1011);
        sp.add_map(i1011, i1100);
        sp.add_map(i1100, i1101);
        sp.add_map(i1101, i1110);
        sp.add_map(i1110, i1111);
        ca.req_symmetry().insert(sp);
        sym_ref.insert(sp);
    }
    }
    //  Fill in random input

    btod_random<4>().perform(bta);
    bta.set_immutable();

    //  Prepare reference data

    dense_tensor<4, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);
    tod_btconv<4>(bta).perform(ta);
    tod_add<4> refop(ta);
    refop.add_op(ta, p1, (symm ? 1.0 : -1.0));
    refop.add_op(ta, p2, (symm ? 1.0 : -1.0));
    refop.add_op(ta, permutation<4>().permute(p1).permute(p2), 1.0);
    refop.perform(true, tb_ref);

    //  Run the symmetrization operation

    btod_copy<4> op_copy(bta);
    btod_symmetrize2<4> sym1(op_copy, p1, symm);
    btod_symmetrize2<4> sym2(sym1, p2, symm);
    sym2.perform(btb);

    tod_btconv<4>(btb).perform(tb);

    //  Compare against the reference: symmetry and data
    block_tensor_ctrl<4, double> ctrlb(btb);

    compare_ref<4>::compare(tns.c_str(), sym2.get_symmetry(), sym_ref);
    compare_ref<4>::compare(tns.c_str(), ctrlb.req_const_symmetry(), sym_ref);

    compare_ref<4>::compare(tns.c_str(), tb, tb_ref, 1e-15);

    } catch(exception &e) {
        if (label) product_table_container::get_instance().erase(tns);

        fail_test(tns.c_str(), __FILE__, __LINE__, e.what());
    } catch(...) {
        if (label) product_table_container::get_instance().erase(tns);

        throw;
    }
    if (label) product_table_container::get_instance().erase(tns);

}


void btod_symmetrize2_test::test_7() {

    const char *testname = "btod_symmetrize2_test::test_7()";

    typedef allocator<double> allocator_t;

    try {

    mask<2> m01, m10, m11;
    m10[0] = true; m01[1] = true;
    m11[0] = true; m11[1] = true;

    mask<4> m0011, m1100, m1111;
    m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

    libtensor::index<2> i2a, i2b;
    i2b[0] = 9; i2b[1] = 19;
    dimensions<2> dims_ia(dimensions<2>(index_range<2>(i2a, i2b)));
    block_index_space<2> bis_ia(dims_ia);
    libtensor::index<4> i4a, i4b;
    i4b[0] = 9; i4b[1] = 9; i4b[2] = 19; i4b[3] = 19;
    dimensions<4> dims_ijab(dimensions<4>(index_range<4>(i4a, i4b)));
    block_index_space<4> bis_ijab(dims_ijab);

    bis_ia.split(m10, 3);
    bis_ia.split(m10, 5);
    bis_ia.split(m10, 8);
    bis_ia.split(m01, 6);
    bis_ia.split(m01, 10);
    bis_ia.split(m01, 16);

    bis_ijab.split(m1100, 3);
    bis_ijab.split(m1100, 5);
    bis_ijab.split(m1100, 8);
    bis_ijab.split(m0011, 6);
    bis_ijab.split(m0011, 10);
    bis_ijab.split(m0011, 16);

    block_tensor<2, double, allocator_t> bt1(bis_ia);
    block_tensor<4, double, allocator_t> bt2(bis_ijab);

    {
        block_tensor_ctrl<2, double> ctrl(bt1);

        libtensor::index<2> i00, i01, i10, i11;
        i10[0] = 1; i01[1] = 1;
        i11[0] = 1; i11[1] = 1;

        se_part<2, double> se(bis_ia, m11, 2);
        se.add_map(i00, i11);
        se.mark_forbidden(i01);
        se.mark_forbidden(i10);
        ctrl.req_symmetry().insert(se);
    }

    btod_random<2>().perform(bt1);
    btod_random<4>().perform(bt2);
    bt1.set_immutable();

    contraction2<2, 2, 0> contr(permutation<4>().permute(1, 2));
    btod_contract2<2, 2, 0> op_contr(contr, bt1, bt1);

    btod_symmetrize2<4>(op_contr, 0, 1, false).perform(bt2);

    dense_tensor<2, double, allocator_t> t1(dims_ia);
    dense_tensor<4, double, allocator_t> ti1(dims_ijab), t2(dims_ijab),
        t2_ref(dims_ijab);

    tod_btconv<2>(bt1).perform(t1);
    tod_btconv<4>(bt2).perform(t2);

    tod_contract2<2, 2, 0>(contr, t1, t1).perform(true, ti1);
    tod_copy<4>(ti1).perform(true, t2_ref);
    tod_copy<4>(ti1, permutation<4>().permute(0, 1), -1.0).
        perform(false, t2_ref);

    compare_ref<4>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
