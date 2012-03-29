#include <libtensor/btod/scalar_transf_double.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/symmetry/so_reduce.h>
#include "../compare_ref.h"
#include "so_reduce_test.h"

namespace libtensor {


void so_reduce_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
}


/** \test Invokes a projection of C1 in 4-space onto 2-space.
        Expects C1 in 2-space.
 **/
void so_reduce_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "so_reduce_test::test_1()";

    try {

    index<4> i1a, i1b;
    i1b[0] = 5; i1b[1] = 5; i1b[2] = 10; i1b[3] = 10;
    index<2> i2a, i2b;
    i2b[0] = 5; i2b[1] = 5;
    dimensions<4> dims1(index_range<4>(i1a, i1b));
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    block_index_space<4> bis1(dims1);
    block_index_space<2> bis2(dims2);

    symmetry<4, double> sym1(bis1);
    symmetry<2, double> sym2(bis2);
    symmetry<2, double> sym2_ref(bis2);
    mask<4> msk; msk[2] = true; msk[3] = true;
    sequence<4, size_t> seq(0);
    index<4> bia, bib; bib[0] = bib[1] = bib[2] = bib[3] = 0;
    index_range<4> bir(bia, bib);
    index_range<4> ir(i1a, i1b);
    so_reduce<4, 2, double> so(sym1, msk, seq, bir, ir);
    so.perform(sym2);

    symmetry<2, double>::iterator i = sym2.begin();
    if(i != sym2.end()) {
        fail_test(testname, __FILE__, __LINE__, "i != sym2.end()");
    }
    compare_ref<2>::compare(testname, sym2, sym2_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


/** \test Invokes a double projection of S5(+) in 5-space onto 2-space.
        Expects S2(+) in 2-space.
 **/
void so_reduce_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "so_reduce_test::test_2()";

    try {

    index<5> i1a, i1b;
    i1b[0] = 5; i1b[1] = 5; i1b[2] = 10; i1b[3] = 8; i1b[4] = 10;
    index<2> i2a, i2b;
    i2b[0] = 5; i2b[1] = 5;
    dimensions<5> dims1(index_range<5>(i1a, i1b));
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    block_index_space<5> bis1(dims1);
    block_index_space<2> bis2(dims2);

    symmetry<5, double> sym1(bis1);
    permutation<5> p1, p2;
    p1.permute(0, 1).permute(1, 2).permute(2, 3).permute(3, 4);
    p2.permute(0, 1);
    scalar_transf<double> tr0, tr1(-1.0);
    sym1.insert(se_perm<5, double>(p1, tr0));
    sym1.insert(se_perm<5, double>(p2, tr0));

    symmetry<2, double> sym2(bis2);
    symmetry<2, double> sym2_ref(bis2);
    mask<5> msk; msk[2] = msk[3] = msk[4] = true;
    sequence<5, size_t> seq(0); seq[3] = 1;
    index<5> bia, bib; bib[0] = bib[1] = bib[2] = bib[3] = bib[4] = 0;
    index_range<5> bir(bia, bib), ir(i1a, i1b);
    so_reduce<5, 3, double>(sym1, msk, seq, bir, ir).perform(sym2);

    sym2_ref.insert(se_perm<2, double>(permutation<2>().permute(0, 1), tr0));

    symmetry<2, double>::iterator i = sym2.begin();
    if(i == sym2.end()) {
        fail_test(testname, __FILE__, __LINE__, "i == sym2.end()");
    }
    compare_ref<2>::compare(testname, sym2, sym2_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Invokes a projection of S2(+) onto 0-space.
 **/
void so_reduce_test::test_3() throw(libtest::test_exception) {

    static const char *testname = "so_reduce_test::test_3()";

    try {

    index<2> i2a, i2b;
    i2b[0] = 5; i2b[1] = 5;
    index<0> i0a, i0b;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<0> dims0(index_range<0>(i0a, i0b));
    block_index_space<2> bis1(dims2);
    block_index_space<0> bis2(dims0);


    symmetry<2, double> sym1(bis1);
    symmetry<0, double> sym2(bis2);
    mask<2> msk; msk[0] = msk[1] = true;
    sequence<2, size_t> seq(0);
    index<2> bia, bib; bib[0] = bib[1] = 0;
    index_range<2> bir(bia, bib), ir(i2a, i2b);
    so_reduce<2, 2, double> so(sym1, msk, seq, bir, ir);
    so.perform(sym2);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

/** \test Invokes a projection of S2(+) onto 2-space.
 **/
void so_reduce_test::test_4() throw(libtest::test_exception) {

    static const char *testname = "so_reduce_test::test_4()";

    try {

    index<2> i2a, i2b;
    i2b[0] = 5; i2b[1] = 5;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    block_index_space<2> bis1(dims2);


    symmetry<2, double> sym1(bis1);
    symmetry<2, double> sym2(bis1);
    mask<2> msk;
    sequence<2, size_t> seq(0);
    index<2> bia, bib; bib[0] = bib[1] = 0;
    index_range<2> bir(bia, bib), ir(i2a, i2b);
    so_reduce<2, 0, double> so(sym1, msk, seq, bir, ir);
    so.perform(sym2);

    symmetry<2, double>::iterator i = sym2.begin();
    if(i == sym2.end()) {
        fail_test(testname, __FILE__, __LINE__, "i == sym2.end()");
    }
    compare_ref<2>::compare(testname, sym2, sym1);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

} // namespace libtensor
