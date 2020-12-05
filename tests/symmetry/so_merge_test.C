#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/symmetry/so_dirsum.h>
#include <libtensor/symmetry/so_merge.h>
#include "../compare_ref.h"
#include "so_merge_test.h"

namespace libtensor {


void so_merge_test::perform() throw(libtest::test_exception) {

    setup_pg_table("cs");

    try {

        test_1();
        test_2();
        test_3();
        //test_4();
        test_5();

    } catch(...) {
        clear_pg_table("cs");
        throw;
    }

    clear_pg_table("cs");
}


/** \test Invokes merge of 2 dimensions of C1 in 4-space onto 3-space.
        Expects C1 in 3-space.
 **/
void so_merge_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "so_merge_test::test_1()";

    try {

    libtensor::index<4> i1a, i1b;
    i1b[0] = 5; i1b[1] = 5; i1b[2] = 10; i1b[3] = 10;
    libtensor::index<3> i2a, i2b;
    i2b[0] = 5; i2b[1] = 5; i2b[2] = 10;
    block_index_space<4> bis1(dimensions<4>(index_range<4>(i1a, i1b)));
    block_index_space<3> bis2(dimensions<3>(index_range<3>(i2a, i2b)));

    symmetry<4, double> sym1(bis1);
    symmetry<3, double> sym2(bis2);
    symmetry<3, double> sym2_ref(bis2);
    mask<4> msk; msk[2] = true; msk[3] = true;
    sequence<4, size_t> seq(0);
    so_merge<4, 1, double>(sym1, msk, seq).perform(sym2);

    symmetry<3, double>::iterator i = sym2.begin();
    if(i != sym2.end()) {
        fail_test(testname, __FILE__, __LINE__, "i != sym2.end()");
    }
    compare_ref<3>::compare(testname, sym2, sym2_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


/** \test Invokes a merge of 3 dim in S5(+) in 5-space onto 3-space.
        Expects S3(+) in 3-space.
 **/
void so_merge_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "so_merge_test::test_2()";

    try {

    libtensor::index<5> i1a, i1b;
    i1b[0] = 5; i1b[1] = 5; i1b[2] = 10; i1b[3] = 10; i1b[4] = 10;
    libtensor::index<3> i2a, i2b;
    i2b[0] = 5; i2b[1] = 5; i2b[2] = 10;
    block_index_space<5> bis1(dimensions<5>(index_range<5>(i1a, i1b)));
    block_index_space<3> bis2(dimensions<3>(index_range<3>(i2a, i2b)));

    symmetry<5, double> sym1(bis1);
    permutation<5> p1a, p1b;
    p1a.permute(0, 1).permute(1, 2).permute(2, 3).permute(3, 4);
    p1b.permute(0, 1);
    scalar_transf<double> tr0;
    sym1.insert(se_perm<5, double>(p1a, tr0));
    sym1.insert(se_perm<5, double>(p1b, tr0));

    symmetry<3, double> sym2(bis2);
    symmetry<3, double> sym2_ref(bis2);
    mask<5> msk;
    msk[2] = true; msk[3] = true; msk[4] = true;
    sequence<5, size_t> seq(0);
    so_merge<5, 2, double>(sym1, msk, seq).perform(sym2);

    permutation<3> p2a, p2b;
    p2a.permute(0, 1).permute(1, 2);
    p2b.permute(0, 1);
    sym2_ref.insert(se_perm<3, double>(p2a, tr0));
    sym2_ref.insert(se_perm<3, double>(p2b, tr0));

    symmetry<3, double>::iterator i = sym2.begin();
    if(i == sym2.end()) {
        fail_test(testname, __FILE__, __LINE__, "i == sym2.end()");
    }
    compare_ref<3>::compare(testname, sym2, sym2_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Invokes merge of 2 dims of S2(+) onto 1-space.
 **/
void so_merge_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "so_merge_test::test_3()";

    try {

    libtensor::index<2> i2a, i2b;
    i2b[0] = 5; i2b[1] = 5;
    libtensor::index<1> i1a, i1b;
    i1b[0] = 5;
    block_index_space<2> bis1(dimensions<2>(index_range<2>(i2a, i2b)));
    block_index_space<1> bis2(dimensions<1>(index_range<1>(i1a, i1b)));

    mask<2> m;
    m[0] = true; m[1] = true;
    sequence<2, size_t> seq(0);
    symmetry<2, double> sym1(bis1);
    symmetry<1, double> sym2(bis2);
    so_merge<2, 1, double>(sym1, m, seq).perform(sym2);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Invokes a projection of S2(+) onto 2-space.
 **/
void so_merge_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "so_merge_test::test_4()";

    try {

    libtensor::index<2> i2a, i2b;
    i2b[0] = 5; i2b[1] = 5;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    block_index_space<2> bis1(dims2);

    mask<2> msk;
    msk[0] = true;
    sequence<2, size_t> seq(0);
    symmetry<2, double> sym1(bis1);
    symmetry<2, double> sym2(bis1);
    so_merge<2, 0, double>(sym1, msk, seq).perform(sym2);

    symmetry<2, double>::iterator i = sym2.begin();
    if(i == sym2.end()) {
        fail_test(testname, __FILE__, __LINE__, "i == sym2.end()");
    }
    compare_ref<2>::compare(testname, sym2, sym1);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Computes the symmetry of the sum of two tensors
 **/
void so_merge_test::test_5() throw(libtest::test_exception) {

    static const char *testname = "so_merge_test::test_5()";

    try {

    point_group_table::label_t ap = 0, app = 1;

    libtensor::index<4> i4a, i4b;
    i4b[0] = 9; i4b[1] = 9; i4b[2] = 9; i4b[3] = 9;
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    block_index_space<4> bis4(dims4);

    mask<4> m1111;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;
    bis4.split(m1111, 2);
    bis4.split(m1111, 5);
    bis4.split(m1111, 7);

    libtensor::index<8> i8a, i8b;
    i8b[0] = 9; i8b[1] = 9; i8b[2] = 9; i8b[3] = 9;
    i8b[4] = 9; i8b[5] = 9; i8b[6] = 9; i8b[7] = 9;
    dimensions<8> dims8(index_range<8>(i8a, i8b));
    block_index_space<8> bis8(dims8);

    mask<8> m11111111;
    m11111111[0] = true; m11111111[1] = true;
    m11111111[2] = true; m11111111[3] = true;
    m11111111[4] = true; m11111111[5] = true;
    m11111111[6] = true; m11111111[7] = true;
    bis8.split(m11111111, 2);
    bis8.split(m11111111, 5);
    bis8.split(m11111111, 7);

    scalar_transf<double> trs(1.0), tras(-1.0);
    se_perm<4, double> sel1(permutation<4>().permute(0, 1).permute(2, 3), trs);
    se_perm<4, double> sel2(permutation<4>().permute(0, 2).permute(1, 3), trs);
    se_perm<4, double> sel3(permutation<4>().permute(2, 3), tras);
    se_label<4, double> sel4(bis4.get_block_index_dims(), "cs");
    block_labeling<4> &bl4 = sel4.get_labeling();
    bl4.assign(m1111, 0, ap);
    bl4.assign(m1111, 1, app);
    bl4.assign(m1111, 2, ap);
    bl4.assign(m1111, 3, app);
    sel4.set_rule(ap);

    symmetry<4, double> sym1(bis4), sym2(bis4), sym3(bis4), sym3_ref(bis4);
    sym1.insert(sel1);
    sym1.insert(sel2);
    sym1.insert(sel3);
    sym1.insert(sel4);
    so_copy<4, double>(sym1).perform(sym2);
    so_copy<4, double>(sym1).perform(sym3_ref);

    symmetry<8, double> symx(bis8);
    so_dirsum<4, 4, double>(sym1, sym2, permutation<8>()).perform(symx);
    sequence<8, size_t> seq(0);
    for(size_t i = 0; i < 4; i++) seq[i] = seq[4 + i] = i;
    so_merge<8, 4, double>(symx, m11111111, seq).perform(sym3);

    compare_ref<4>::compare(testname, sym3, sym3_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor
