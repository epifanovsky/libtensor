#include <libtensor/core/block_index_subspace_builder.h>
#include "block_index_subspace_builder_test.h"

namespace libtensor {


void block_index_subspace_builder_test::perform()
    throw(libtest::test_exception) {

    test_0();
    test_1();
    test_2();
    test_3();
    test_4();
}


/** \test Subspace of [6(2),6(3)] with mask [1,1] (no dimensions removed).
        Expected result: [6(2),2(3)].
 **/
void block_index_subspace_builder_test::test_0()
    throw(libtest::test_exception) {

    static const char *testname =
        "block_index_subspace_builder_test::test_0()";

    try {

    index<2> i2a, i2b;

    i2b[0] = 5; i2b[1] = 5;

    dimensions<2> dims2(index_range<2>(i2a, i2b));

    block_index_space<2> bis(dims2);
    mask<2> m2a, m2b;
    m2a[0] = true; m2b[1] = true;
    bis.split(m2a, 2);
    bis.split(m2b, 3);
    block_index_space<2> bis_ref(bis);

    mask<2> m;
    m[0] = true; m[1] = true;

    block_index_subspace_builder<2, 0> bb(bis, m);
    if(!bis_ref.equals(bb.get_bis())) {
        fail_test(testname, __FILE__, __LINE__,
            "!bis_ref.equals(bb.get_bis()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Subspace of [6,6] with mask [1,0]. No blocks.
        Expected result: [6].
 **/
void block_index_subspace_builder_test::test_1()
    throw(libtest::test_exception) {

    static const char *testname =
        "block_index_subspace_builder_test::test_1()";

    try {

    index<2> i2a, i2b;
    index<1> i1a, i1b;

    i2b[0] = 5; i2b[1] = 5;
    i1b[0] = 5;

    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<1> dims1(index_range<1>(i1a, i1b));

    block_index_space<2> bis(dims2);
    block_index_space<1> bis_ref(dims1);
    mask<2> m;
    m[0] = true;

    block_index_subspace_builder<1, 1> bb(bis, m);
    if(!bis_ref.equals(bb.get_bis())) {
        fail_test(testname, __FILE__, __LINE__,
            "!bis_ref.equals(bb.get_bis()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Subspace of [6,10] with mask [1,0]. No blocks.
        Expected result: [6].
 **/
void block_index_subspace_builder_test::test_2()
    throw(libtest::test_exception) {

    static const char *testname =
        "block_index_subspace_builder_test::test_2()";

    try {

    index<2> i2a, i2b;
    index<1> i1a, i1b;

    i2b[0] = 5; i2b[1] = 9;
    i1b[0] = 5;

    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<1> dims1(index_range<1>(i1a, i1b));

    block_index_space<2> bis(dims2);
    block_index_space<1> bis_ref(dims1);
    mask<2> m;
    m[0] = true;

    block_index_subspace_builder<1, 1> bb(bis, m);
    if(!bis_ref.equals(bb.get_bis())) {
        fail_test(testname, __FILE__, __LINE__,
            "!bis_ref.equals(bb.get_bis()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Subspace of [6(2),6(3),6(2),6(3)] with mask [1,0,1,0].
        Expected result: [6(2),6(2)].
 **/
void block_index_subspace_builder_test::test_3()
    throw(libtest::test_exception) {

    static const char *testname =
        "block_index_subspace_builder_test::test_3()";

    try {

    index<4> i4a, i4b;
    index<2> i2a, i2b;

    i4b[0] = 5; i4b[1] = 5; i4b[2] = 5; i4b[3] = 5;
    i2b[0] = 5; i2b[1] = 5;

    dimensions<4> dims4(index_range<4>(i4a, i4b));
    dimensions<2> dims2(index_range<2>(i2a, i2b));

    block_index_space<4> bis(dims4);
    block_index_space<2> bis_ref(dims2);

    mask<4> m4a, m4b;
    mask<2> m2;
    m4a[0] = true; m4b[1] = true; m4a[2] = true; m4b[3] = true;
    m2[0] = true; m2[1] = true;
    bis.split(m4a, 2);
    bis.split(m4b, 3);
    bis_ref.split(m2, 2);

    mask<4> m;
    m[0] = true; m[2] = true;

    block_index_subspace_builder<2, 2> bb(bis, m);
    if(!bis_ref.equals(bb.get_bis())) {
        fail_test(testname, __FILE__, __LINE__,
            "!bis_ref.equals(bb.get_bis()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Subspace of [6(2),6(3),6(2),6(3)] with mask [1,0,0,1].
        Expected result: [6(2),6(3)].
 **/
void block_index_subspace_builder_test::test_4()
    throw(libtest::test_exception) {

    static const char *testname =
        "block_index_subspace_builder_test::test_4()";

    try {

    index<4> i4a, i4b;
    index<2> i2a, i2b;

    i4b[0] = 5; i4b[1] = 5; i4b[2] = 5; i4b[3] = 5;
    i2b[0] = 5; i2b[1] = 5;

    dimensions<4> dims4(index_range<4>(i4a, i4b));
    dimensions<2> dims2(index_range<2>(i2a, i2b));

    block_index_space<4> bis(dims4);
    block_index_space<2> bis_ref(dims2);

    mask<4> m4a, m4b;
    mask<2> m2a, m2b;
    m4a[0] = true; m4b[1] = true; m4a[2] = true; m4b[3] = true;
    m2a[0] = true; m2b[1] = true;
    bis.split(m4a, 2);
    bis.split(m4b, 3);
    bis_ref.split(m2a, 2);
    bis_ref.split(m2b, 3);

    mask<4> m;
    m[0] = true; m[3] = true;

    block_index_subspace_builder<2, 2> bb(bis, m);
    if(!bis_ref.equals(bb.get_bis())) {
        fail_test(testname, __FILE__, __LINE__,
            "!bis_ref.equals(bb.get_bis()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
