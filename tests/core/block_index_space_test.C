#include <sstream>
#include <libtensor/core/print_dimensions.h>
#include <libtensor/core/block_index_space.h>
#include "block_index_space_test.h"

namespace libtensor {

void block_index_space_test::perform() throw(libtest::test_exception) {

    test_ctor_1();

    //  test_split_* do not use equals() for comparison
    test_split_1();
    test_split_2();
    test_split_3();
    test_split_4();

    //  test_equals_* use split()
    test_equals_1();
    test_equals_2();
    test_equals_3();
    test_equals_4();
    test_equals_5();

    //  test_match_* use split() and equals()
    test_match_1();
    test_match_2();
    test_match_3();
    test_match_4();
    test_match_5();

    //  test_permute_* use split() and equals()
    test_permute_1();

    test_exc_1();
    test_exc_2();

}


void block_index_space_test::test_ctor_1() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_ctor_1()";

    try {

    index<4> i1, i2;
    i2[0] = 8; i2[1] = 8; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);

    if(bis.get_type(0) != bis.get_type(1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Invalid initial splitting type (1).");
    }
    if(bis.get_type(2) != bis.get_type(3)) {
        fail_test(testname, __FILE__, __LINE__,
            "Invalid initial splitting type (2).");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void block_index_space_test::test_split_1() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_split_1()";

    try {

    index<1> i_0;
    index<1> i_1; i_1[0] = 1;
    index<1> i_2; i_2[0] = 2;
    index<1> i_3; i_3[0] = 3;
    index<1> i_4; i_4[0] = 4;
    index<1> i_5; i_5[0] = 5;
    index<1> i_7; i_7[0] = 7;
    index<1> i_9; i_9[0] = 9;

    dimensions<1> d_1(index_range<1>(i_0, i_0));
    dimensions<1> d_2(index_range<1>(i_0, i_1));
    dimensions<1> d_3(index_range<1>(i_0, i_2));
    dimensions<1> d_5(index_range<1>(i_0, i_4));
    dimensions<1> d_8(index_range<1>(i_0, i_7));
    dimensions<1> d_10(index_range<1>(i_0, i_9));

    block_index_space<1> bis(d_10);

    if(!bis.get_dims().equals(d_10)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect total dimensions");
    }
    if(!bis.get_block_index_dims().equals(d_1)) {
        std::ostringstream ss;
        ss << "(1) Incorrect block index dimensions: "
            << bis.get_block_index_dims() << " vs. "
            << d_1 << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!bis.get_block_start(i_0).equals(i_0)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [0] start");
    }
    if(!bis.get_block_dims(i_0).equals(d_10)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [0] dimensions");
    }

    mask<1> splmsk; splmsk[0] = true;
    bis.split(splmsk, 2);

    if(!bis.get_dims().equals(d_10)) {
        fail_test(testname, __FILE__, __LINE__,
            "(2) Incorrect total dimensions");
    }
    if(!bis.get_block_index_dims().equals(d_2)) {
        std::ostringstream ss;
        ss << "(2) Incorrect block index dimensions: "
            << bis.get_block_index_dims() << " vs. "
            << d_2 << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!bis.get_block_start(i_0).equals(i_0)) {
        std::ostringstream ss;
        ss << "(2) Incorrect start of block " << i_0 << ": "
            << bis.get_block_start(i_0) << " vs. "
            << i_0 << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!bis.get_block_dims(i_0).equals(d_2)) {
        std::ostringstream ss;
        ss << "(2) Incorrect dimensions of block " << i_0 << ": "
            << bis.get_block_dims(i_0) << " vs. "
            << d_2 << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!bis.get_block_start(i_1).equals(i_2)) {
        std::ostringstream ss;
        ss << "(2) Incorrect start of block " << i_1 << ": "
            << bis.get_block_start(i_1) << " vs. "
            << i_2 << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!bis.get_block_dims(i_1).equals(d_8)) {
        std::ostringstream ss;
        ss << "(2) Incorrect dimensions of block " << i_1 << ": "
            << bis.get_block_dims(i_1) << " vs. "
            << d_8 << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    bis.split(splmsk, 5);

    if(!bis.get_dims().equals(d_10)) {
        fail_test(testname, __FILE__, __LINE__,
            "(3) Incorrect total dimensions");
    }
    if(!bis.get_block_index_dims().equals(d_3)) {
        std::ostringstream ss;
        ss << "(3) Incorrect block index dimensions: "
            << bis.get_block_index_dims() << " vs. "
            << d_3 << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!bis.get_block_start(i_0).equals(i_0)) {
        fail_test(testname, __FILE__, __LINE__,
            "(3) Incorrect block [0] start");
    }
    if(!bis.get_block_dims(i_0).equals(d_2)) {
        fail_test(testname, __FILE__, __LINE__,
            "(3) Incorrect block [0] dimensions");
    }
    if(!bis.get_block_start(i_1).equals(i_2)) {
        fail_test(testname, __FILE__, __LINE__,
            "(3) Incorrect block [1] start");
    }
    if(!bis.get_block_dims(i_1).equals(d_3)) {
        fail_test(testname, __FILE__, __LINE__,
            "(3) Incorrect block [1] dimensions");
    }
    if(!bis.get_block_start(i_2).equals(i_5)) {
        fail_test(testname, __FILE__, __LINE__,
            "(3) Incorrect block [2] start");
    }
    if(!bis.get_block_dims(i_2).equals(d_5)) {
        fail_test(testname, __FILE__, __LINE__,
            "(3) Incorrect block [2] dimensions");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

void block_index_space_test::test_split_2() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_split_2()";

    try {

    index<1> i_0;
    index<1> i_1; i_1[0] = 1;
    index<1> i_2; i_2[0] = 2;
    index<1> i_3; i_3[0] = 3;

    dimensions<1> d_1(index_range<1>(i_0, i_0));
    dimensions<1> d_2(index_range<1>(i_0, i_1));
    dimensions<1> d_3(index_range<1>(i_0, i_2));

    block_index_space<1> bis(d_3);

    if(!bis.get_dims().equals(d_3)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect total dimensions");
    }

    mask<1> splmsk; splmsk[0] = true;
    bis.split(splmsk, 1);
    bis.split(splmsk, 2);

    if(!bis.get_dims().equals(d_3)) {
        fail_test(testname, __FILE__, __LINE__,
            "(2) Incorrect total dimensions");
    }
    if(!bis.get_block_index_dims().equals(d_3)) {
        fail_test(testname, __FILE__, __LINE__,
            "(2) Incorrect block index dimensions");
    }
    if(!bis.get_block_start(i_0).equals(i_0)) {
        fail_test(testname, __FILE__, __LINE__,
            "(2) Incorrect block [0] start");
    }
    if(!bis.get_block_dims(i_0).equals(d_1)) {
        fail_test(testname, __FILE__, __LINE__,
            "(2) Incorrect block [0] dimensions");
    }
    if(!bis.get_block_start(i_1).equals(i_1)) {
        fail_test(testname, __FILE__, __LINE__,
            "(2) Incorrect block [1] start");
    }
    if(!bis.get_block_dims(i_1).equals(d_1)) {
        fail_test(testname, __FILE__, __LINE__,
            "(2) Incorrect block [1] dimensions");
    }
    if(!bis.get_block_start(i_2).equals(i_2)) {
        fail_test(testname, __FILE__, __LINE__,
            "(2) Incorrect block [2] start");
    }
    if(!bis.get_block_dims(i_2).equals(d_1)) {
        fail_test(testname, __FILE__, __LINE__,
            "(2) Incorrect block [2] dimensions");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

void block_index_space_test::test_split_3() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_split_3()";

    try {

    index<2> i_00;
    index<2> i_01; i_01[1] = 1;
    index<2> i_02; i_02[1] = 2;
    index<2> i_03; i_03[1] = 3;
    index<2> i_10; i_10[0] = 1;
    index<2> i_11; i_11[0] = 1; i_11[1] = 1;
    index<2> i_12; i_12[0] = 1; i_12[1] = 2;
    index<2> i_13; i_13[0] = 1; i_13[1] = 3;
    index<2> i_20; i_20[0] = 2;
    index<2> i_21; i_21[0] = 2; i_21[1] = 1;
    index<2> i_22; i_22[0] = 2; i_22[1] = 2;
    index<2> i_23; i_23[0] = 2; i_23[1] = 3;
    index<2> i_30; i_30[0] = 3;
    index<2> i_31; i_31[0] = 3; i_31[1] = 1;
    index<2> i_32; i_32[0] = 3; i_32[1] = 2;
    index<2> i_33; i_33[0] = 3; i_33[1] = 3;
    index<2> i_55; i_55[0] = 5; i_55[1] = 5;

    dimensions<2> d_11(index_range<2>(i_00, i_00));
    dimensions<2> d_12(index_range<2>(i_00, i_01));
    dimensions<2> d_13(index_range<2>(i_00, i_02));
    dimensions<2> d_21(index_range<2>(i_00, i_10));
    dimensions<2> d_22(index_range<2>(i_00, i_11));
    dimensions<2> d_23(index_range<2>(i_00, i_12));
    dimensions<2> d_31(index_range<2>(i_00, i_20));
    dimensions<2> d_32(index_range<2>(i_00, i_21));
    dimensions<2> d_33(index_range<2>(i_00, i_22));
    dimensions<2> d_66(index_range<2>(i_00, i_55));

    block_index_space<2> bis(d_66);

    size_t type_6 = bis.get_type(0);
    if(bis.get_type(1) != type_6) {
        fail_test(testname, __FILE__, __LINE__,
            "Incorrect dimension type.");
    }

    mask<2> splmsk; splmsk[0] = true; splmsk[1] = true;
    bis.split(splmsk, 1);
    bis.split(splmsk, 3);

    if(!bis.get_dims().equals(d_66)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect total dimensions");
    }
    if(!bis.get_block_index_dims().equals(d_33)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block index dimensions");
    }
    size_t typ = bis.get_type(0);
    if(bis.get_type(1) != typ) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect splitting type");
    }
    if(!bis.get_block_start(i_00).equals(i_00)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [0,0] start");
    }
    if(!bis.get_block_dims(i_00).equals(d_11)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [0,0] dimensions");
    }
    if(!bis.get_block_start(i_01).equals(i_01)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [0,1] start");
    }
    if(!bis.get_block_dims(i_01).equals(d_12)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [0,1] dimensions");
    }
    if(!bis.get_block_start(i_02).equals(i_03)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [0,2] start");
    }
    if(!bis.get_block_dims(i_02).equals(d_13)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [0,2] dimensions");
    }
    if(!bis.get_block_start(i_10).equals(i_10)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [1,0] start");
    }
    if(!bis.get_block_dims(i_10).equals(d_21)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [1,0] dimensions");
    }
    if(!bis.get_block_start(i_11).equals(i_11)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [1,1] start");
    }
    if(!bis.get_block_dims(i_11).equals(d_22)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [1,1] dimensions");
    }
    if(!bis.get_block_start(i_12).equals(i_13)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [1,2] start");
    }
    if(!bis.get_block_dims(i_12).equals(d_23)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [1,2] dimensions");
    }
    if(!bis.get_block_start(i_20).equals(i_30)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [2,0] start");
    }
    if(!bis.get_block_dims(i_20).equals(d_31)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [2,0] dimensions");
    }
    if(!bis.get_block_start(i_21).equals(i_31)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [2,1] start");
    }
    if(!bis.get_block_dims(i_21).equals(d_32)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [2,1] dimensions");
    }
    if(!bis.get_block_start(i_22).equals(i_33)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [2,2] start");
    }
    if(!bis.get_block_dims(i_22).equals(d_33)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block [2,2] dimensions");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


void block_index_space_test::test_split_4() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_split_4()";

    try {

    index<2> i_00;
    index<2> i_11; i_11[0] = 1; i_11[1] = 1;
    index<2> i_55; i_55[0] = 5; i_55[1] = 5;
    dimensions<2> d_22(index_range<2>(i_00, i_11));
    dimensions<2> d_66(index_range<2>(i_00, i_55));
    mask<2> msk1, msk2;
    msk1[0] = true;
    msk2[1] = true;

    block_index_space<2> bis(d_66);
    bis.split(msk1, 2);
    bis.split(msk2, 2);

    if(!bis.get_dims().equals(d_66)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect total dimensions");
    }
    if(!bis.get_block_index_dims().equals(d_22)) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Incorrect block index dimensions");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


void block_index_space_test::test_equals_1() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_equals_1()";

    try {

    index<1> i_0;
    index<1> i_9; i_9[0] = 9;

    dimensions<1> d_10(index_range<1>(i_0, i_9));

    block_index_space<1> bis1(d_10), bis2(d_10);
    mask<1> splmsk; splmsk[0] = true;

    if(!bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (1a) failed");
    }
    if(!bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (1b) failed");
    }

    bis1.split(splmsk, 2);
    if(bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (2a) failed");
    }
    if(bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (2b) failed");
    }

    bis2.split(splmsk, 2);
    if(!bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (3a) failed");
    }
    if(!bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (3b) failed");
    }

    bis1.split(splmsk, 8);
    bis2.split(splmsk, 4);
    if(bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (4a) failed");
    }
    if(bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (4b) failed");
    }

    bis1.split(splmsk, 6);
    bis2.split(splmsk, 6);
    if(bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (5a) failed");
    }
    if(bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (5b) failed");
    }

    bis1.split(splmsk, 4);
    bis2.split(splmsk, 8);
    if(!bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (6a) failed");
    }
    if(!bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (6b) failed");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void block_index_space_test::test_equals_2() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_equals_2()";

    try {

    index<2> i_00;
    index<2> i_99; i_99[0] = 9; i_99[1] = 9;

    dimensions<2> dims(index_range<2>(i_00, i_99));

    block_index_space<2> bis1(dims), bis2(dims), bis3(dims), bis4(dims);

    if(!bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (1a) failed");
    }
    if(!bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (1b) failed");
    }

    mask<2> msk; msk[0] = true; msk[1] = true;

    bis1.split(msk, 2);
    bis1.split(msk, 4);
    bis1.split(msk, 6);
    bis1.split(msk, 8);
    if(bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (2a) failed");
    }
    if(bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (2b) failed");
    }

    bis2.split(msk, 8);
    bis2.split(msk, 6);
    bis2.split(msk, 4);
    bis2.split(msk, 2);
    if(!bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (3a) failed");
    }
    if(!bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (3b) failed");
    }

    mask<2> msk1, msk2; msk1[0] = true; msk2[1] = true;
    bis3.split(msk1, 3);
    bis3.split(msk1, 7);
    if(bis4.equals(bis3)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (4a) failed");
    }
    if(bis3.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (4b) failed");
    }

    bis4.split(msk2, 7);
    bis4.split(msk2, 3);
    if(bis4.equals(bis3)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (5a) failed");
    }
    if(bis3.equals(bis4)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (5b) failed");
    }

    permutation<2> p1;
    p1.permute(0, 1);
    bis4.permute(p1);
    if(!bis4.equals(bis3)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (6a) failed");
    }
    if(!bis3.equals(bis4)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (6b) failed");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void block_index_space_test::test_equals_3() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_equals_3()";

    try {

    permutation<4> perm;
    perm.permute(0, 2).permute(1, 3);
    index<4> i_00;
    index<4> i_89;
    i_89[0] = 8; i_89[1] = 8; i_89[2] = 9; i_89[3] = 9;
    mask<4> msk1, msk2;
    msk1[0] = true; msk1[1] = true;
    msk2[2] = true; msk2[3] = true;

    dimensions<4> dims_89(index_range<4>(i_00, i_89));
    dimensions<4> dims_98(dims_89); dims_98.permute(perm);

    block_index_space<4> bis1(dims_89), bis2(dims_98);

    if(bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (1a) failed");
    }
    if(bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (1b) failed");
    }

    bis1.split(msk1, 5);
    bis2.split(msk2, 5);
    if(bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (2a) failed");
    }
    if(bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (2b) failed");
    }

    bis1.split(msk2, 3);
    bis2.split(msk1, 3);
    if(bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (3a) failed");
    }
    if(bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (3b) failed");
    }

    bis2.permute(perm);
    if(!bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (4a) failed");
    }
    if(!bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (4b) failed");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void block_index_space_test::test_equals_4() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_equals_4()";

    try {

    index<4> i1, i2;
    i2[0] = 8; i2[1] = 8; i2[2] = 9; i2[3] = 9;
    mask<4> m1100, m0011, m0010, m0001;
    m1100[0] = true; m1100[1] = true;
    m0011[2] = true; m0011[3] = true;
    m0010[2] = true;
    m0001[3] = true;

    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis1(dims), bis2(dims);

    if(!bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (1a) failed");
    }
    if(!bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (1b) failed");
    }

    bis1.split(m1100, 4);
    bis1.split(m0011, 5);
    bis2.split(m1100, 4);
    bis2.split(m0010, 6);
    bis2.split(m0001, 6);
    if(bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (2a) failed");
    }
    if(bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (2b) failed");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void block_index_space_test::test_equals_5() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_equals_5()";

    try {

    index<4> i1, i2;
    i2[0] = 8; i2[1] = 8; i2[2] = 9; i2[3] = 9;
    mask<4> m1100, m0011, m0010, m0001;
    m1100[0] = true; m1100[1] = true;
    m0011[2] = true; m0011[3] = true;
    m0010[2] = true;
    m0001[3] = true;

    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis1(dims), bis2(dims);

    if(!bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (1a) failed");
    }
    if(!bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (1b) failed");
    }

    bis1.split(m1100, 4);
    bis1.split(m0011, 5);
    bis2.split(m1100, 4);
    bis2.split(m0010, 5);
    bis2.split(m0001, 5);
    if(bis1.equals(bis2)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (2a) failed");
    }
    if(bis2.equals(bis1)) {
        fail_test(testname, __FILE__, __LINE__,
            "Equality test (2b) failed");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void block_index_space_test::test_match_1() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_match_1()";

    try {

    index<4> i1, i2;
    i2[0] = 8; i2[1] = 8; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims), bis_ref(dims);

    bis.match_splits();

    if(!bis.equals(bis_ref)) {
        fail_test(testname, __FILE__, __LINE__, "Invalid result.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void block_index_space_test::test_match_2() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_match_2()";

    try {

    index<4> i1, i2;
    i2[0] = 8; i2[1] = 9; i2[2] = 10; i2[3] = 11;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims), bis_ref(dims);
    mask<4> m1000, m0100, m0010, m0001;
    m1000[0] = true; m0100[1] = true; m0010[2] = true; m0001[3] = true;

    bis.split(m1000, 5);
    bis_ref.split(m1000, 5);
    bis.split(m0100, 5);
    bis_ref.split(m0100, 5);
    bis.split(m0010, 5);
    bis_ref.split(m0010, 5);
    bis.split(m0001, 5);
    bis_ref.split(m0001, 5);

    bis.match_splits();

    if(!bis.equals(bis_ref)) {
        fail_test(testname, __FILE__, __LINE__, "Invalid result.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void block_index_space_test::test_match_3() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_match_3()";

    try {

    index<4> i1, i2;
    i2[0] = 8; i2[1] = 8; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims), bis_ref(dims);
    mask<4> m1100, m0011, m0010, m0001;
    m1100[0] = true; m1100[1] = true;
    m0011[2] = true; m0011[3] = true;

    bis.split(m1100, 5);
    bis_ref.split(m1100, 5);
    bis.split(m0011, 5);
    bis_ref.split(m0011, 5);

    bis.match_splits();

    if(!bis.equals(bis_ref)) {
        fail_test(testname, __FILE__, __LINE__, "Invalid result.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void block_index_space_test::test_match_4() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_match_4()";

    try {

    index<4> i1, i2;
    i2[0] = 8; i2[1] = 8; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims), bis_ref(dims);
    mask<4> m1100, m0011, m0010, m0001;
    m1100[0] = true; m1100[1] = true;
    m0011[2] = true; m0011[3] = true;
    m0010[2] = true;
    m0001[3] = true;

    bis.split(m1100, 5);
    bis.split(m0010, 5);
    bis.split(m0001, 5);
    bis_ref.split(m1100, 5);
    bis_ref.split(m0011, 5);

    bis.match_splits();

    if(!bis.equals(bis_ref)) {
        fail_test(testname, __FILE__, __LINE__, "Invalid result.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void block_index_space_test::test_match_5() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_match_5()";

    try {

    index<4> i1, i2;
    i2[0] = 8; i2[1] = 8; i2[2] = 8; i2[3] = 8;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims), bis_ref(dims);
    mask<4> m1000, m0100, m0011, m1111;
    m1000[0] = true; m0100[1] = true; m0011[2] = true; m0011[3] = true;
    m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

    bis.split(m1000, 5);
    bis.split(m0100, 5);
    bis.split(m0011, 5);
    bis_ref.split(m1111, 5);

    bis.match_splits();

    if(!bis.equals(bis_ref)) {
        fail_test(testname, __FILE__, __LINE__, "Invalid result.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void block_index_space_test::test_permute_1() throw(libtest::test_exception) {

    static const char *testname =
        "block_index_space_test::test_permute_1()";

    try {

    permutation<4> perm; perm.permute(1, 2);
    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 11; i2[3] = 11;
    dimensions<4> dimsa(index_range<4>(i1, i2));
    block_index_space<4> bisa(dimsa);
    dimensions<4> dimsb_ref(dimsa); dimsb_ref.permute(perm);
    block_index_space<4> bisb_ref(dimsb_ref);

    mask<4> msk1, msk2, msk3, msk4;
    msk1[0] = true; msk1[1] = true;
    msk2[2] = true; msk2[3] = true;
    msk3[0] = true; msk3[2] = true;
    msk4[1] = true; msk4[3] = true;

    bisa.split(msk1, 3);
    bisa.split(msk1, 5);
    bisa.split(msk2, 4);
    bisb_ref.split(msk3, 3);
    bisb_ref.split(msk3, 5);
    bisb_ref.split(msk4, 4);

    dimensions<4> bidimsa(bisa.get_block_index_dims());
    dimensions<4> bidimsb_ref(bidimsa); bidimsb_ref.permute(perm);
    block_index_space<4> bisb(bisa);
    bisb.permute(perm);
    dimensions<4> bidimsb(bisb.get_block_index_dims());

    if(!bisb.equals(bisb_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Incorrect permuted block index space (1).");
    }
    if(!bidimsb.equals(bidimsb_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Incorrect block index dimensions (1).");
    }

    block_index_space<4> bisb2(bisb);
    dimensions<4> bidimsb2(bisb2.get_block_index_dims());

    if(!bisb2.equals(bisb_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Incorrect permuted block index space (2).");
    }
    if(!bidimsb2.equals(bidimsb_ref)) {
        fail_test(testname, __FILE__, __LINE__,
            "Incorrect block index dimensions (2).");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void block_index_space_test::test_exc_1() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_exc_1()";

    try {

    index<1> i_0;
    index<1> i_1; i_1[0] = 1;
    index<1> i_2; i_2[0] = 2;
    index<1> i_3; i_3[0] = 3;
    index<1> i_4; i_4[0] = 4;
    index<1> i_5; i_5[0] = 5;
    index<1> i_7; i_7[0] = 7;
    index<1> i_9; i_9[0] = 9;

    dimensions<1> d_1(index_range<1>(i_0, i_0));
    dimensions<1> d_2(index_range<1>(i_0, i_1));
    dimensions<1> d_3(index_range<1>(i_0, i_2));
    dimensions<1> d_5(index_range<1>(i_0, i_4));
    dimensions<1> d_8(index_range<1>(i_0, i_7));
    dimensions<1> d_10(index_range<1>(i_0, i_9));

    block_index_space<1> bis(d_10);
    bool ok;

#ifdef LIBTENSOR_DEBUG
    ok = false;
    try {
        bis.get_block_start(i_1);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Index out of bounds in get_block_start()");
    }

    ok = false;
    try {
        bis.get_block_dims(i_1);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Index out of bounds in get_block_dims()");
    }

    ok = false;
    try {
        bis.get_block_start(i_2);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(2) Index out of bounds in get_block_start()");
    }

    ok = false;
    try {
        bis.get_block_dims(i_2);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(2) Index out of bounds in get_block_dims()");
    }
#endif // LIBTENSOR_DEBUG

    mask<1> splmsk; splmsk[0] = true;
    bis.split(splmsk, 5);

#ifdef LIBTENSOR_DEBUG
    ok = false;
    try {
        bis.get_block_start(i_2);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(3) Index out of bounds in get_block_start()");
    }

    ok = false;
    try {
        bis.get_block_dims(i_2);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(3) Index out of bounds in get_block_dims()");
    }

    ok = false;
    try {
        bis.get_block_start(i_3);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(4) Index out of bounds in get_block_start()");
    }

    ok = false;
    try {
        bis.get_block_dims(i_3);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(4) Index out of bounds in get_block_dims()");
    }
#endif // LIBTENSOR_DEBUG

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}

void block_index_space_test::test_exc_2() throw(libtest::test_exception) {

    static const char *testname = "block_index_space_test::test_exc_2()";

    try {

    index<2> i_00;
    index<2> i_01; i_01[1] = 1;
    index<2> i_02; i_02[1] = 2;
    index<2> i_03; i_03[1] = 3;
    index<2> i_10; i_10[0] = 1;
    index<2> i_11; i_11[0] = 1; i_11[1] = 1;
    index<2> i_12; i_12[0] = 1; i_12[1] = 2;
    index<2> i_13; i_13[0] = 1; i_13[1] = 3;
    index<2> i_14; i_14[0] = 1; i_14[1] = 4;
    index<2> i_20; i_20[0] = 2;
    index<2> i_21; i_21[0] = 2; i_21[1] = 1;
    index<2> i_22; i_22[0] = 2; i_22[1] = 2;
    index<2> i_23; i_23[0] = 2; i_23[1] = 3;
    index<2> i_30; i_30[0] = 3;
    index<2> i_31; i_31[0] = 3; i_31[1] = 1;
    index<2> i_32; i_32[0] = 3; i_32[1] = 2;
    index<2> i_33; i_33[0] = 3; i_33[1] = 3;
    index<2> i_41; i_41[0] = 4; i_41[1] = 1;
    index<2> i_55; i_55[0] = 5; i_55[1] = 5;

    dimensions<2> d_11(index_range<2>(i_00, i_00));
    dimensions<2> d_12(index_range<2>(i_00, i_01));
    dimensions<2> d_13(index_range<2>(i_00, i_02));
    dimensions<2> d_21(index_range<2>(i_00, i_10));
    dimensions<2> d_22(index_range<2>(i_00, i_11));
    dimensions<2> d_23(index_range<2>(i_00, i_12));
    dimensions<2> d_31(index_range<2>(i_00, i_20));
    dimensions<2> d_32(index_range<2>(i_00, i_21));
    dimensions<2> d_33(index_range<2>(i_00, i_22));
    dimensions<2> d_66(index_range<2>(i_00, i_55));

    block_index_space<2> bis(d_66);
    bool ok;

#ifdef LIBTENSOR_DEBUG
    ok = false;
    try {
        bis.get_block_start(i_11);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Index out of bounds in get_block_start()");
    }

    ok = false;
    try {
        bis.get_block_dims(i_11);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(1) Index out of bounds in get_block_dims()");
    }
#endif // LIBTENSOR_DEBUG

    mask<2> splmsk1, splmsk2;
    splmsk1[0] = true; splmsk2[1] = true;
    bis.split(splmsk1, 1);

#ifdef LIBTENSOR_DEBUG
    ok = false;
    try {
        bis.get_block_start(i_21);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(2) Index out of bounds in get_block_start()");
    }

    ok = false;
    try {
        bis.get_block_dims(i_21);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(2) Index out of bounds in get_block_dims()");
    }
#endif // LIBTENSOR_DEBUG

    bis.split(splmsk1, 3);

#ifdef LIBTENSOR_DEBUG
    ok = false;
    try {
        bis.get_block_start(i_31);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(3) Index out of bounds in get_block_start()");
    }

    ok = false;
    try {
        bis.get_block_dims(i_31);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(3) Index out of bounds in get_block_dims()");
    }

    ok = false;
    try {
        bis.get_block_start(i_33);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(4) Index out of bounds in get_block_start()");
    }

    ok = false;
    try {
        bis.get_block_dims(i_33);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(4) Index out of bounds in get_block_dims()");
    }
#endif // LIBTENSOR_DEBUG

    bis.split(splmsk2, 1);

#ifdef LIBTENSOR_DEBUG
    ok = false;
    try {
        bis.get_block_start(i_32);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(5) Index out of bounds in get_block_start()");
    }

    ok = false;
    try {
        bis.get_block_dims(i_32);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(5) Index out of bounds in get_block_dims()");
    }
#endif // LIBTENSOR_DEBUG

    bis.split(splmsk2, 3);

#ifdef LIBTENSOR_DEBUG
    ok = false;
    try {
        bis.get_block_start(i_33);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(6) Index out of bounds in get_block_start()");
    }

    ok = false;
    try {
        bis.get_block_dims(i_33);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(6) Index out of bounds in get_block_dims()");
    }

    ok = false;
    try {
        bis.get_block_start(i_14);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(7) Index out of bounds in get_block_start()");
    }

    ok = false;
    try {
        bis.get_block_dims(i_14);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(7) Index out of bounds in get_block_dims()");
    }

    ok = false;
    try {
        bis.get_block_start(i_41);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(8) Index out of bounds in get_block_start()");
    }

    ok = false;
    try {
        bis.get_block_dims(i_41);
    } catch(out_of_bounds &exc) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "(8) Index out of bounds in get_block_dims()");
    }
#endif // LIBTENSOR_DEBUG

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor
