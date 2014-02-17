#include <libtensor/expr/bispace/bispace.h>
#include "bispace_test.h"

namespace libtensor {

void bispace_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
    test_4();
    test_5();
    test_6();
    test_7();
    test_8();
    test_9();
}

void bispace_test::test_1() throw(libtest::test_exception) {

    //
    //  Simple 1-d spaces
    //

    const char *testname = "bispace_test::test_1()";

    try {

    bispace<1> a(10), b(10);
    b.split(4);

    //  Make references

    index<1> i1, i2;
    i2[0] = 9;
    dimensions<1> dims(index_range<1>(i1, i2));

    block_index_space<1> bisa_ref(dims), bisb_ref(dims);
    mask<1> mskb; mskb[0] = true;
    bisb_ref.split(mskb, 4);

    mask<1> sma0_ref, smb0_ref;
    sma0_ref[0] = true;
    smb0_ref[0] = true;

    //  Run tests

    if(!a.get_bis().equals(bisa_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for a.");
    }
    if(!a.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for a.");
    }
    if(!b.get_bis().equals(bisb_ref)) {
        fail_test(testname, __FILE__, __LINE__, ""
            "Bis test failed for b.");
    }
    if(!b.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for b.");
    }

    bispace<1> ca(a), cb(b);

    if(!ca.get_bis().equals(bisa_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for ca.");
    }
    if(!ca.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for ca.");
    }
    if(!cb.get_bis().equals(bisb_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for cb.");
    }
    if(!cb.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cb.");
    }

    if(!a.equals(ca)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for ca.");
    }
    if(!b.equals(cb)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for cb.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void bispace_test::test_2() throw(libtest::test_exception) {

    //
    //  Simple 2-d spaces
    //

    const char *testname = "bispace_test::test_2()";

    try {

    bispace<1> a(10), b(10);
    b.split(4);
    bispace<2> aa(a|a), bb(b|b);

    //  Make references

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));

    block_index_space<2> bisa_ref(dims), bisb_ref(dims);
    mask<2> mskb1, mskb2;
    mskb1[0] = true; mskb2[1] = true;
    bisb_ref.split(mskb1, 4);
    bisb_ref.split(mskb2, 4);

    mask<2> sma0_ref, sma1_ref, smb0_ref, smb1_ref;
    sma0_ref[0] = true; sma1_ref[1] = true;
    smb0_ref[0] = true; smb1_ref[1] = true;

    //  Run tests

    if(!aa.get_bis().equals(bisa_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for aa.");
    }
    if(!aa.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for aa.");
    }
    if(!aa.get_sym_mask(1).equals(sma1_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for aa.");
    }
    if(!bb.get_bis().equals(bisb_ref)) {
        fail_test(testname, __FILE__, __LINE__, ""
            "Bis test failed for bb.");
    }
    if(!bb.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for bb.");
    }
    if(!bb.get_sym_mask(1).equals(smb1_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for bb.");
    }

    bispace<2> caa(aa), cbb(bb);

    if(!caa.get_bis().equals(bisa_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for caa.");
    }
    if(!caa.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for caa.");
    }
    if(!caa.get_sym_mask(1).equals(sma1_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for caa.");
    }
    if(!cbb.get_bis().equals(bisb_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for cbb.");
    }
    if(!cbb.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cbb.");
    }
    if(!cbb.get_sym_mask(1).equals(smb1_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cbb.");
    }

    if(!aa.equals(caa)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for caa.");
    }
    if(!bb.equals(cbb)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for cbb.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_test::test_3() throw(libtest::test_exception) {

    //
    //  2-d spaces with symmetry between 1-d
    //

    const char *testname = "bispace_test::test_3()";

    try {

    bispace<1> a(10), b(10);
    b.split(4);
    bispace<2> aa(a&a), bb(b&b);

    //  Make references

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));

    block_index_space<2> bisa_ref(dims), bisb_ref(dims);
    mask<2> mskb;
    mskb[0] = true; mskb[1] = true;
    bisb_ref.split(mskb, 4);

    mask<2> sma0_ref, smb0_ref;
    sma0_ref[0] = true; sma0_ref[1] = true;
    smb0_ref[0] = true; smb0_ref[1] = true;

    //  Run tests

    if(!aa.get_bis().equals(bisa_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for aa.");
    }
    if(!aa.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for aa.");
    }
    if(!aa.get_sym_mask(1).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for aa.");
    }
    if(!bb.get_bis().equals(bisb_ref)) {
        fail_test(testname, __FILE__, __LINE__, ""
            "Bis test failed for bb.");
    }
    if(!bb.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for bb.");
    }
    if(!bb.get_sym_mask(1).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for bb.");
    }

    bispace<2> caa(aa), cbb(bb);

    if(!caa.get_bis().equals(bisa_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for caa.");
    }
    if(!caa.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for caa.");
    }
    if(!caa.get_sym_mask(1).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for caa.");
    }
    if(!cbb.get_bis().equals(bisb_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for cbb.");
    }
    if(!cbb.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cbb.");
    }
    if(!cbb.get_sym_mask(1).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cbb.");
    }

    if(!aa.equals(caa)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for caa.");
    }
    if(!bb.equals(cbb)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for cbb.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_test::test_4() throw(libtest::test_exception) {

    //
    //  4-d spaces with symmetry between symmetric 2-d
    //

    const char *testname = "bispace_test::test_4()";

    try {

    bispace<1> a(10), b(10);
    b.split(4);
    bispace<2> aa(a&a), bb(b&b);
    bispace<4> aaaa(aa&aa), bbbb(bb&bb);

    //  Make references

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));

    block_index_space<4> bisa_ref(dims), bisb_ref(dims);
    mask<4> mskb;
    mskb[0] = true; mskb[1] = true; mskb[2] = true; mskb[3] = true;
    bisb_ref.split(mskb, 4);

    mask<4> sma0_ref, smb0_ref;
    sma0_ref[0] = true; sma0_ref[1] = true;
    sma0_ref[2] = true; sma0_ref[3] = true;
    smb0_ref[0] = true; smb0_ref[1] = true;
    smb0_ref[2] = true; smb0_ref[3] = true;

    //  Run tests

    if(!aaaa.get_bis().equals(bisa_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for aaaa.");
    }
    if(!aaaa.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for aaaa.");
    }
    if(!aaaa.get_sym_mask(1).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for aaaa.");
    }
    if(!aaaa.get_sym_mask(2).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for aaaa.");
    }
    if(!aaaa.get_sym_mask(3).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for aaaa.");
    }
    if(!bbbb.get_bis().equals(bisb_ref)) {
        fail_test(testname, __FILE__, __LINE__, ""
            "Bis test failed for bbbb.");
    }
    if(!bbbb.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for bbbb.");
    }
    if(!bbbb.get_sym_mask(1).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for bbbb.");
    }
    if(!bbbb.get_sym_mask(2).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for bbbb.");
    }
    if(!bbbb.get_sym_mask(3).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for bbbb.");
    }

    bispace<4> caaaa(aaaa), cbbbb(bbbb);

    if(!caaaa.get_bis().equals(bisa_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for caaaa.");
    }
    if(!caaaa.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for caaaa.");
    }
    if(!caaaa.get_sym_mask(1).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for caaaa.");
    }
    if(!caaaa.get_sym_mask(2).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for caaaa.");
    }
    if(!caaaa.get_sym_mask(3).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for caaaa.");
    }
    if(!cbbbb.get_bis().equals(bisb_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for cbbbb.");
    }
    if(!cbbbb.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cbbbb.");
    }
    if(!cbbbb.get_sym_mask(1).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cbbbb.");
    }
    if(!cbbbb.get_sym_mask(2).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cbbbb.");
    }
    if(!cbbbb.get_sym_mask(3).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cbbbb.");
    }

    if(!aaaa.equals(caaaa)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for caaaa.");
    }
    if(!bbbb.equals(cbbbb)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for cbbbb.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_test::test_5() throw(libtest::test_exception) {

    //
    //  4-d spaces with symmetry made of 1-d
    //

    const char *testname = "bispace_test::test_5()";

    try {

    bispace<1> a(20), b(20), i(10), j(10);
    bispace<4> ijab1(i&j|a&b);
    a.split(10).split(15);
    b.split(10).split(15);
    i.split(4);
    j.split(4);
    bispace<4> ijab2(i&j|a&b);

    //  Make references

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 19; i2[3] = 19;
    dimensions<4> dims(index_range<4>(i1, i2));

    block_index_space<4> bis1_ref(dims), bis2_ref(dims);
    mask<4> msk2_1, msk2_2;
    msk2_1[0] = true; msk2_1[1] = true;
    msk2_2[2] = true; msk2_2[3] = true;
    bis2_ref.split(msk2_1, 4);
    bis2_ref.split(msk2_2, 10);
    bis2_ref.split(msk2_2, 15);

    mask<4> sma0_ref, smb0_ref, sma2_ref, smb2_ref;
    sma0_ref[0] = true; sma0_ref[1] = true;
    sma2_ref[2] = true; sma2_ref[3] = true;
    smb0_ref[0] = true; smb0_ref[1] = true;
    smb2_ref[2] = true; smb2_ref[3] = true;

    //  Run tests

    if(!ijab1.get_bis().equals(bis1_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for ijab1.");
    }
    if(!ijab1.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for ijab1.");
    }
    if(!ijab1.get_sym_mask(1).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for ijab1.");
    }
    if(!ijab1.get_sym_mask(2).equals(sma2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for ijab1.");
    }
    if(!ijab1.get_sym_mask(3).equals(sma2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for ijab1.");
    }
    if(!ijab2.get_bis().equals(bis2_ref)) {
        fail_test(testname, __FILE__, __LINE__, ""
            "Bis test failed for ijab2.");
    }
    if(!ijab2.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for ijab2.");
    }
    if(!ijab2.get_sym_mask(1).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for ijab2.");
    }
    if(!ijab2.get_sym_mask(2).equals(smb2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for ijab2.");
    }
    if(!ijab2.get_sym_mask(3).equals(smb2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for ijab2.");
    }

    bispace<4> cijab1(ijab1), cijab2(ijab2);

    if(!cijab1.get_bis().equals(bis1_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for cijab1.");
    }
    if(!cijab1.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cijab1.");
    }
    if(!cijab1.get_sym_mask(1).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cijab1.");
    }
    if(!cijab1.get_sym_mask(2).equals(sma2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for cijab1.");
    }
    if(!cijab1.get_sym_mask(3).equals(sma2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for cijab1.");
    }
    if(!cijab2.get_bis().equals(bis2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for cijab2.");
    }
    if(!ijab2.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cijab2.");
    }
    if(!cijab2.get_sym_mask(1).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cijab2.");
    }
    if(!cijab2.get_sym_mask(2).equals(smb2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cijab2.");
    }
    if(!cijab2.get_sym_mask(3).equals(smb2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cijab2.");
    }

    if(!ijab1.equals(cijab1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for cijab1.");
    }
    if(!ijab2.equals(cijab2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for cijab2.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_test::test_6() throw(libtest::test_exception) {

    //
    //  4-d spaces with symmetry made of 2-d,
    //  equivalence to the same made of 1-d
    //

    const char *testname = "bispace_test::test_6()";

    try {

    bispace<1> a(20), b(20), i(10), j(10);
    bispace<2> ab1(a&b), ij1(i&j);
    bispace<4> ijab1_ref(i&j|a&b);
    bispace<4> ijab1(ij1|ab1);
    a.split(10).split(15);
    b.split(10).split(15);
    i.split(4);
    j.split(4);
    bispace<2> ab2(a&b), ij2(i&j);
    bispace<4> ijab2_ref(i&j|a&b);
    bispace<4> ijab2(ij2|ab2);

    //  Make references

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 19; i2[3] = 19;
    dimensions<4> dims(index_range<4>(i1, i2));

    block_index_space<4> bis1_ref(dims), bis2_ref(dims);
    mask<4> msk2_1, msk2_2;
    msk2_1[0] = true; msk2_1[1] = true;
    msk2_2[2] = true; msk2_2[3] = true;
    bis2_ref.split(msk2_1, 4);
    bis2_ref.split(msk2_2, 10);
    bis2_ref.split(msk2_2, 15);

    mask<4> sma0_ref, smb0_ref, sma2_ref, smb2_ref;
    sma0_ref[0] = true; sma0_ref[1] = true;
    sma2_ref[2] = true; sma2_ref[3] = true;
    smb0_ref[0] = true; smb0_ref[1] = true;
    smb2_ref[2] = true; smb2_ref[3] = true;

    //  Run tests

    if(!ijab1.get_bis().equals(bis1_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for ijab1.");
    }
    if(!ijab1.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for ijab1.");
    }
    if(!ijab1.get_sym_mask(1).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for ijab1.");
    }
    if(!ijab1.get_sym_mask(2).equals(sma2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for ijab1.");
    }
    if(!ijab1.get_sym_mask(3).equals(sma2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for ijab1.");
    }
    if(!ijab2.get_bis().equals(bis2_ref)) {
        fail_test(testname, __FILE__, __LINE__, ""
            "Bis test failed for ijab2.");
    }
    if(!ijab2.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for ijab2.");
    }
    if(!ijab2.get_sym_mask(1).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for ijab2.");
    }
    if(!ijab2.get_sym_mask(2).equals(smb2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for ijab2.");
    }
    if(!ijab2.get_sym_mask(3).equals(smb2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for ijab2.");
    }

    bispace<4> cijab1(ijab1), cijab2(ijab2);

    if(!cijab1.get_bis().equals(bis1_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for cijab1.");
    }
    if(!cijab1.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cijab1.");
    }
    if(!cijab1.get_sym_mask(1).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cijab1.");
    }
    if(!cijab1.get_sym_mask(2).equals(sma2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for cijab1.");
    }
    if(!cijab1.get_sym_mask(3).equals(sma2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for cijab1.");
    }
    if(!cijab2.get_bis().equals(bis2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for cijab2.");
    }
    if(!ijab2.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cijab2.");
    }
    if(!cijab2.get_sym_mask(1).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cijab2.");
    }
    if(!cijab2.get_sym_mask(2).equals(smb2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cijab2.");
    }
    if(!cijab2.get_sym_mask(3).equals(smb2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cijab2.");
    }

    if(!ijab1.equals(cijab1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for cijab1.");
    }
    if(!ijab2.equals(cijab2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for cijab2.");
    }

    if(!ijab1.equals(ijab1_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for cijab1_ref.");
    }
    if(!ijab2.equals(ijab2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for cijab2_ref.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_test::test_7() throw(libtest::test_exception) {

    //
    //  4-d spaces with symmetry made of 1-d, reordered
    //

    const char *testname = "bispace_test::test_7()";

    try {

    bispace<1> a(20), b(20), i(10), j(10);
    bispace<4> ijab1(i|a|j|b, i&j|a&b);
    a.split(10).split(15);
    b.split(10).split(15);
    i.split(4);
    j.split(4);
    bispace<4> ijab2(i|a|j|b, i&j|a&b);

    //  Make references

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 19; i2[2] = 9; i2[3] = 19;
    dimensions<4> dims(index_range<4>(i1, i2));

    block_index_space<4> bis1_ref(dims), bis2_ref(dims);
    mask<4> msk2_1, msk2_2;
    msk2_1[0] = true; msk2_1[2] = true;
    msk2_2[1] = true; msk2_2[3] = true;
    bis2_ref.split(msk2_1, 4);
    bis2_ref.split(msk2_2, 10);
    bis2_ref.split(msk2_2, 15);

    mask<4> sma0_ref, smb0_ref, sma2_ref, smb2_ref;
    sma0_ref[0] = true; sma0_ref[2] = true;
    sma2_ref[1] = true; sma2_ref[3] = true;
    smb0_ref[0] = true; smb0_ref[2] = true;
    smb2_ref[1] = true; smb2_ref[3] = true;

    //  Run tests

    if(!ijab1.get_bis().equals(bis1_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for ijab1.");
    }
    if(!ijab1.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for ijab1.");
    }
    if(!ijab1.get_sym_mask(1).equals(sma2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for ijab1.");
    }
    if(!ijab1.get_sym_mask(2).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for ijab1.");
    }
    if(!ijab1.get_sym_mask(3).equals(sma2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for ijab1.");
    }
    if(!ijab2.get_bis().equals(bis2_ref)) {
        fail_test(testname, __FILE__, __LINE__, ""
            "Bis test failed for ijab2.");
    }
    if(!ijab2.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for ijab2.");
    }
    if(!ijab2.get_sym_mask(1).equals(smb2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for ijab2.");
    }
    if(!ijab2.get_sym_mask(2).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for ijab2.");
    }
    if(!ijab2.get_sym_mask(3).equals(smb2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for ijab2.");
    }

    bispace<4> cijab1(ijab1), cijab2(ijab2);

    if(!cijab1.get_bis().equals(bis1_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for cijab1.");
    }
    if(!cijab1.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cijab1.");
    }
    if(!cijab1.get_sym_mask(1).equals(sma2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cijab1.");
    }
    if(!cijab1.get_sym_mask(2).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for cijab1.");
    }
    if(!cijab1.get_sym_mask(3).equals(sma2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for cijab1.");
    }
    if(!cijab2.get_bis().equals(bis2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for cijab2.");
    }
    if(!ijab2.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cijab2.");
    }
    if(!cijab2.get_sym_mask(1).equals(smb2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cijab2.");
    }
    if(!cijab2.get_sym_mask(2).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cijab2.");
    }
    if(!cijab2.get_sym_mask(3).equals(smb2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cijab2.");
    }

    if(!ijab1.equals(cijab1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for cijab1.");
    }
    if(!ijab2.equals(cijab2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for cijab2.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_test::test_8() throw(libtest::test_exception) {

    //
    //  5-d spaces with symmetry made of mixed dims, reordered
    //

    const char *testname = "bispace_test::test_8()";

    try {

    bispace<1> a(20), b(20), i(10);
    bispace<2> ab1(a&b), cd1(a&b);
    bispace<5> abicd1(ab1|i|cd1, ab1&cd1|i);
    a.split(10).split(15);
    b.split(10).split(15);
    i.split(4);
    bispace<2> ab2(a&b), cd2(a&b);
    bispace<5> abicd2(ab2|i|cd2, ab2&cd2|i);

    //  Make references

    index<5> i1, i2;
    i2[0] = 19; i2[1] = 19; i2[2] = 9; i2[3] = 19; i2[4] = 19;
    dimensions<5> dims(index_range<5>(i1, i2));

    block_index_space<5> bis1_ref(dims), bis2_ref(dims);
    mask<5> msk2_1, msk2_2;
    msk2_1[0] = true; msk2_1[1] = true; msk2_1[3] = true; msk2_1[4] = true;
    msk2_2[2] = true;
    bis2_ref.split(msk2_1, 10);
    bis2_ref.split(msk2_1, 15);
    bis2_ref.split(msk2_2, 4);

    mask<5> sm0_ref, sm2_ref;
    sm0_ref[0] = true; sm0_ref[1] = true;
    sm0_ref[3] = true; sm0_ref[4] = true;
    sm2_ref[2] = true;

    //  Run tests

    if(!abicd1.get_bis().equals(bis1_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for abicd1.");
    }
    if(!abicd1.get_sym_mask(0).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for abicd1.");
    }
    if(!abicd1.get_sym_mask(1).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for abicd1.");
    }
    if(!abicd1.get_sym_mask(2).equals(sm2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for abicd1.");
    }
    if(!abicd1.get_sym_mask(3).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for abicd1.");
    }
    if(!abicd1.get_sym_mask(4).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 4 test failed for abicd1.");
    }
    if(!abicd2.get_bis().equals(bis2_ref)) {
        fail_test(testname, __FILE__, __LINE__, ""
            "Bis test failed for abicd2.");
    }
    if(!abicd2.get_sym_mask(0).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for abicd2.");
    }
    if(!abicd2.get_sym_mask(1).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for abicd2.");
    }
    if(!abicd2.get_sym_mask(2).equals(sm2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for abicd2.");
    }
    if(!abicd2.get_sym_mask(3).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for abicd2.");
    }
    if(!abicd2.get_sym_mask(4).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 4 test failed for abicd2.");
    }

    bispace<5> cabicd1(abicd1), cabicd2(abicd2);

    if(!cabicd1.get_bis().equals(bis1_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for cabicd1.");
    }
    if(!cabicd1.get_sym_mask(0).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cabicd1.");
    }
    if(!cabicd1.get_sym_mask(1).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cabicd1.");
    }
    if(!cabicd1.get_sym_mask(2).equals(sm2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for cabicd1.");
    }
    if(!cabicd1.get_sym_mask(3).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for cabicd1.");
    }
    if(!cabicd1.get_sym_mask(4).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 4 test failed for cabicd1.");
    }
    if(!cabicd2.get_bis().equals(bis2_ref)) {
        fail_test(testname, __FILE__, __LINE__, ""
            "Bis test failed for cabicd2.");
    }
    if(!cabicd2.get_sym_mask(0).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cabicd2.");
    }
    if(!cabicd2.get_sym_mask(1).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cabicd2.");
    }
    if(!cabicd2.get_sym_mask(2).equals(sm2_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for cabicd2.");
    }
    if(!cabicd2.get_sym_mask(3).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for cabicd2.");
    }
    if(!cabicd2.get_sym_mask(4).equals(sm0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 4 test failed for cabicd2.");
    }

    if(!abicd1.equals(cabicd1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for cabicd1.");
    }
    if(!abicd2.equals(cabicd2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for cabicd2.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_test::test_9() throw(libtest::test_exception) {

    //
    //  4-d spaces with symmetry between 1-d
    //

    const char *testname = "bispace_test::test_9()";

    try {

    bispace<1> a(10), b(10);
    b.split(4);
    bispace<4> aaaa(a&a&a&a), bbbb(b&b&b&b);

    //  Make references

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));

    block_index_space<4> bisa_ref(dims), bisb_ref(dims);
    mask<4> mskb;
    mskb[0] = true; mskb[1] = true; mskb[2] = true; mskb[3] = true;
    bisb_ref.split(mskb, 4);

    mask<4> sma0_ref, smb0_ref;
    sma0_ref[0] = true; sma0_ref[1] = true;
    sma0_ref[2] = true; sma0_ref[3] = true;

    smb0_ref[0] = true; smb0_ref[1] = true;
    smb0_ref[2] = true; smb0_ref[3] = true;

    //  Run tests

    if(!aaaa.get_bis().equals(bisa_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for aaaa.");
    }
    if(!aaaa.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for aaaa.");
    }
    if(!aaaa.get_sym_mask(1).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for aaaa.");
    }
    if(!aaaa.get_sym_mask(2).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for aaaa.");
    }
    if(!aaaa.get_sym_mask(3).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for aaaa.");
    }
    if(!bbbb.get_bis().equals(bisb_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for bbbb.");
    }
    if(!bbbb.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for bbbb.");
    }
    if(!bbbb.get_sym_mask(1).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for bbbb.");
    }
    if(!bbbb.get_sym_mask(2).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for bbbb.");
    }
    if(!bbbb.get_sym_mask(3).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for bbbb.");
    }

    bispace<4> caaaa(aaaa), cbbbb(bbbb);

    if(!caaaa.get_bis().equals(bisa_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for caaaa.");
    }
    if(!caaaa.get_sym_mask(0).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for caaaa.");
    }
    if(!caaaa.get_sym_mask(1).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for caaaa.");
    }
    if(!caaaa.get_sym_mask(2).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for caaaa.");
    }
    if(!caaaa.get_sym_mask(3).equals(sma0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for caaaa.");
    }
    if(!cbbbb.get_bis().equals(bisb_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Bis test failed for cbbbb.");
    }
    if(!cbbbb.get_sym_mask(0).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 0 test failed for cbbbb.");
    }
    if(!cbbbb.get_sym_mask(1).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 1 test failed for cbbbb.");
    }
    if(!cbbbb.get_sym_mask(2).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 2 test failed for cbbbb.");
    }
    if(!cbbbb.get_sym_mask(3).equals(smb0_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Symmetry mask 3 test failed for cbbbb.");
    }

    if(!aaaa.equals(caaaa)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for caaaa.");
    }
    if(!bbbb.equals(cbbbb)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test failed for cbbbb.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

