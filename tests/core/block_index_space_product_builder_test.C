#include <libtensor/core/block_index_space_product_builder.h>
#include "block_index_space_product_builder_test.h"

namespace libtensor {


void block_index_space_product_builder_test::perform()
    throw(libtest::test_exception) {

    test_0a();
    test_0b();
    test_1a();
    test_1b();
    test_2a();
    test_2b();
    test_3a();
    test_3b();
}


/** \test Product of spaces [6(2)] and []. Expected result: [6(2)].
 **/
void block_index_space_product_builder_test::test_0a()
    throw(libtest::test_exception) {

    static const char *testname =
        "block_index_space_product_builder_test::test_0a()";

    try {

    index<1> i1a, i1b;
    index<0> i0a, i0b;
    i1b[0] = 5;

    dimensions<1> dimsa(index_range<1>(i1a, i1b));
    dimensions<0> dimsb(index_range<0>(i0a, i0b));

    block_index_space<1> bisa(dimsa);
    block_index_space<0> bisb(dimsb);
    mask<1> ma;
    ma[0] = true;
    bisa.split(ma, 2);
    bisa.split(ma, 3);
    block_index_space<1> bis_ref(bisa);

    permutation<1> p;
    block_index_space_product_builder<1, 0> bb(bisa, bisb, p);
    if(!bis_ref.equals(bb.get_bis())) {
        fail_test(testname, __FILE__, __LINE__,
            "!bis_ref.equals(bb.get_bis()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Product of spaces [] and [6(2)]. Expected result: [6(2)].
 **/
void block_index_space_product_builder_test::test_0b()
    throw(libtest::test_exception) {

    static const char *testname =
        "block_index_space_product_builder_test::test_0b()";

    try {

    index<1> i1a, i1b;
    index<0> i0a, i0b;
    i1b[0] = 5;

    dimensions<0> dimsa(index_range<0>(i0a, i0b));
    dimensions<1> dimsb(index_range<1>(i1a, i1b));

    block_index_space<0> bisa(dimsa);
    block_index_space<1> bisb(dimsb);
    mask<1> mb;
    mb[0] = true;
    bisb.split(mb, 2);
    block_index_space<1> bis_ref(bisb);

    permutation<1> p;
    block_index_space_product_builder<0, 1> bb(bisa, bisb, p);
    if(!bis_ref.equals(bb.get_bis())) {
        fail_test(testname, __FILE__, __LINE__,
            "!bis_ref.equals(bb.get_bis()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Product space of [6] and [5]. No blocks. Expected result: [6, 5].
 **/
void block_index_space_product_builder_test::test_1a()
    throw(libtest::test_exception) {

    static const char *testname =
        "block_index_space_product_builder_test::test_1a()";

    try {

    index<1> i1a, i1b, i2a, i2b;
    index<2> i3a, i3b;

    i1b[0] = 5; i2b[0] = 4;
    i3b[0] = 5; i3b[1] = 4;

    dimensions<1> dimsa(index_range<1>(i1a, i1b));
    dimensions<1> dimsb(index_range<1>(i2a, i2b));
    dimensions<2> dimsc(index_range<2>(i3a, i3b));

    block_index_space<1> bisa(dimsa), bisb(dimsb);
    block_index_space<2> bis_ref(dimsc);

    permutation<2> p;
    block_index_space_product_builder<1, 1> bb(bisa, bisb, p);
    if(!bis_ref.equals(bb.get_bis())) {
        fail_test(testname, __FILE__, __LINE__,
            "!bis_ref.equals(bb.get_bis()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Product space of [6] and [5]. No blocks. Permutation.
        Expected result: [5, 6].
 **/
void block_index_space_product_builder_test::test_1b()
    throw(libtest::test_exception) {

    static const char *testname =
        "block_index_space_product_builder_test::test_1b()";

    try {

    index<1> i1a, i1b, i2a, i2b;
    index<2> i3a, i3b;

    i1b[0] = 5; i2a[0] = 4;
    i3b[0] = 4; i3b[1] = 5;

    dimensions<1> dimsa(index_range<1>(i1a, i1b));
    dimensions<1> dimsb(index_range<1>(i2a, i2b));
    dimensions<2> dimsc(index_range<2>(i3a, i3b));

    block_index_space<1> bisa(dimsa), bisb(dimsb);
    block_index_space<2> bis_ref(dimsc);

    permutation<2> p; p.permute(0, 1);
    block_index_space_product_builder<1, 1> bb(bisa, bisb, p);
    if(!bis_ref.equals(bb.get_bis())) {
        fail_test(testname, __FILE__, __LINE__,
            "!bis_ref.equals(bb.get_bis()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Product space of [6(2), 6(2)] and [9(3)].
        Expected result: [6(2),6(2),9(3)].
 **/
void block_index_space_product_builder_test::test_2a()
    throw(libtest::test_exception) {

    static const char *testname =
        "block_index_space_product_builder_test::test_2a()";

    try {

    index<1> i1a, i1b;
    index<2> i2a, i2b;
    index<3> i3a, i3b;

    i1b[0] = 8;
    i2b[0] = 5; i2b[1] = 5;
    i3b[0] = 5; i3b[1] = 5; i3b[2] = 8;

    dimensions<2> dimsa(index_range<2>(i2a, i2b));
    dimensions<1> dimsb(index_range<1>(i1a, i1b));
    dimensions<3> dimsc(index_range<3>(i3a, i3b));

    block_index_space<2> bisa(dimsa);
    block_index_space<1> bisb(dimsb);
    block_index_space<3> bis_ref(dimsc);

    mask<2> ma;
    ma[0] = true; ma[1] = true;
    mask<1> mb;
    mb[0] = true;
    mask<3> mc1, mc2;
    mc1[0] = true; mc1[1] = true;
    mc2[2] = true;

    bisa.split(ma, 2);
    bisb.split(mb, 3);
    bisb.split(mb, 6);
    bis_ref.split(mc1, 2);
    bis_ref.split(mc2, 3);
    bis_ref.split(mc2, 6);

    permutation<3> p;
    block_index_space_product_builder<2, 1> bb(bisa, bisb, p);
    if(!bis_ref.equals(bb.get_bis())) {
        fail_test(testname, __FILE__, __LINE__,
            "!bis_ref.equals(bb.get_bis()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Product space of [6(2), 6(2)] and [9(3)]. Permutation.
        Expected result: [6(2),9(3),6(2)].
 **/
void block_index_space_product_builder_test::test_2b()
    throw(libtest::test_exception) {

    static const char *testname =
        "block_index_space_product_builder_test::test_2b()";

    try {

    index<1> i1a, i1b;
    index<2> i2a, i2b;
    index<3> i3a, i3b;

    i1b[0] = 8;
    i2b[0] = 5; i2b[1] = 5;
    i3b[0] = 5; i3b[1] = 8; i3b[2] = 5;

    dimensions<2> dimsa(index_range<2>(i2a, i2b));
    dimensions<1> dimsb(index_range<1>(i1a, i1b));
    dimensions<3> dimsc(index_range<3>(i3a, i3b));

    block_index_space<2> bisa(dimsa);
    block_index_space<1> bisb(dimsb);
    block_index_space<3> bis_ref(dimsc);

    mask<2> ma;
    ma[0] = true; ma[1] = true;
    mask<1> mb;
    mb[0] = true;
    mask<3> mc1, mc2;
    mc1[0] = true; mc1[2] = true;
    mc2[1] = true;

    bisa.split(ma, 2);
    bisb.split(mb, 3);
    bisb.split(mb, 6);
    bis_ref.split(mc1, 2);
    bis_ref.split(mc2, 3);
    bis_ref.split(mc2, 6);

    permutation<3> p; p.permute(0, 1).permute(1, 2);
    block_index_space_product_builder<2, 1> bb(bisa, bisb, p);
    if(!bis_ref.equals(bb.get_bis())) {
        fail_test(testname, __FILE__, __LINE__,
            "!bis_ref.equals(bb.get_bis()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Product space of [6(2),6(3)] and [6(2),6(3)].
        Expected result: [6(2),6(3),6(2),6(3)].
 **/
void block_index_space_product_builder_test::test_3a()
    throw(libtest::test_exception) {

    static const char *testname =
        "block_index_space_product_builder_test::test_3a()";

    try {

    index<2> i2a, i2b;
    index<4> i4a, i4b;

    i2b[0] = 5; i2b[1] = 5;
    i4b[0] = 5; i4b[1] = 5; i4b[2] = 5; i4b[3] = 5;

    dimensions<2> dimsa(index_range<2>(i2a, i2b));
    dimensions<4> dimsc(index_range<4>(i4a, i4b));

    block_index_space<2> bisa(dimsa);
    block_index_space<4> bis_ref(dimsc);

    mask<2> ma1, ma2;
    ma1[0] = true; ma2[1] = true;
    bisa.split(ma1, 2);
    bisa.split(ma2, 2);
    bisa.split(ma2, 3);

    mask<4> mc1, mc2;
    mc1[0] = true; mc1[2] = true;
    mc2[1] = true; mc2[3] = true;
    bis_ref.split(mc1, 2);
    bis_ref.split(mc2, 2);
    bis_ref.split(mc2, 3);

    permutation<4> p;
    block_index_space_product_builder<2, 2> bb(bisa, bisa, p);
    if(!bis_ref.equals(bb.get_bis())) {
        fail_test(testname, __FILE__, __LINE__,
            "!bis_ref.equals(bb.get_bis()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

/** \test Product space of [6(2),6(3)] and [6(2),6(3)].
        Expected result: [6(2),6(3),6(2),6(3)].
 **/
void block_index_space_product_builder_test::test_3b()
    throw(libtest::test_exception) {

    static const char *testname =
        "block_index_space_product_builder_test::test_3b()";

    try {

    index<2> i2a, i2b;
    index<4> i4a, i4b;

    i2b[0] = 5; i2b[1] = 5;
    i4b[0] = 5; i4b[1] = 5; i4b[2] = 5; i4b[3] = 5;

    dimensions<2> dimsa(index_range<2>(i2a, i2b));
    dimensions<4> dimsc(index_range<4>(i4a, i4b));

    block_index_space<2> bisa(dimsa);
    block_index_space<4> bis_ref(dimsc);

    mask<2> ma1, ma2;
    ma1[0] = true; ma2[1] = true;
    bisa.split(ma1, 2);
    bisa.split(ma2, 2);
    bisa.split(ma2, 3);

    mask<4> mc1, mc2;
    mc1[0] = true; mc1[1] = true;
    mc2[2] = true; mc2[3] = true;
    bis_ref.split(mc1, 2);
    bis_ref.split(mc2, 2);
    bis_ref.split(mc2, 3);

    permutation<4> p; p.permute(1, 2);
    block_index_space_product_builder<2, 2> bb(bisa, bisa, p);
    if(!bis_ref.equals(bb.get_bis())) {
        fail_test(testname, __FILE__, __LINE__,
            "!bis_ref.equals(bb.get_bis()");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
