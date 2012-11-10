#include <libtensor/core/abs_index.h>
#include "abs_index_test.h"

namespace libtensor {


void abs_index_test::perform() throw(libtest::test_exception) {

    test_ctor_1();
    test_ctor_2();
    test_ctor_3();
    test_ctor_4();
    test_ctor_5();
    test_inc_1();
    test_inc_2();
    test_inc_3();
    test_inc_4();
    test_last_1();
    test_get_index_1();
}


void abs_index_test::test_ctor_1() throw(libtest::test_exception) {

    static const char *testname = "abs_index_test::test_ctor_1()";

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    index<2> i;
    i[0] = 0; i[1] = 0;
    abs_index<2> ai(i, dims);

    if(ai.get_abs_index() != 0) {
        fail_test(testname, __FILE__, __LINE__,
            "abs(0,0) in (10,10) doesn't return 0.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void abs_index_test::test_ctor_2() throw(libtest::test_exception) {

    static const char *testname = "abs_index_test::test_ctor_2()";

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    index<2> i;
    i[0] = 1; i[1] = 0;
    abs_index<2> ai(i, dims);

    if(ai.get_abs_index() != 10) {
        fail_test(testname, __FILE__, __LINE__,
            "abs(1,0) in (10,10) doesn't return 10.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void abs_index_test::test_ctor_3() throw(libtest::test_exception) {

    static const char *testname = "abs_index_test::test_ctor_3()";

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    index<2> i;
    i[0] = 9; i[1] = 9;
    abs_index<2> ai(i, dims);

    if(ai.get_abs_index() != 99) {
        fail_test(testname, __FILE__, __LINE__,
            "abs(9,9) in (10,10) doesn't return 99.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void abs_index_test::test_ctor_4() throw(libtest::test_exception) {

    static const char *testname = "abs_index_test::test_ctor_4()";

    try {

    index<4> i1, i2;
    i2[0] = 1; i2[1] = 4; i2[2] = 1; i2[3] = 13;
    dimensions<4> dims(index_range<4>(i1, i2));
    index<4> i;
    i[0] = 1; i[1] = 0; i[2] = 1; i[3] = 0;
    abs_index<4> ai(154, dims);

    if(!i.equals(ai.get_index())) {
        fail_test(testname, __FILE__, __LINE__,
            "abs(154) in (2,5,2,14) doesn't return (1,0,1,0).");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void abs_index_test::test_ctor_5() throw(libtest::test_exception) {

    static const char *testname = "abs_index_test::test_ctor_5()";

    try {

    index<4> i1, i2;
    i2[0] = 1; i2[1] = 1; i2[2] = 1; i2[3] = 1;
    dimensions<4> dims(index_range<4>(i1, i2));
    index<4> i;
    i[0] = 2; i[1] = 2; i[2] = 2; i[3] = 2;

    bool ok = false;
    try {
        abs_index<4> ai(i, dims);
    } catch(out_of_bounds&) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__, "out_of_bounds expected.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void abs_index_test::test_inc_1() throw(libtest::test_exception) {

    static const char *testname = "abs_index_test::test_inc_1()";

    try {

    index<4> i1, i2;
    i2[0] = 1; i2[1] = 1; i2[2] = 1; i2[3] = 1;
    dimensions<4> dims(index_range<4>(i1, i2));
    index<4> i;
    abs_index<4> ai(i, dims);

    if(!ai.inc()) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(0,0,0,0) doesn't return true.");
    }
    i[0] = 0; i[1] = 0; i[2] = 0; i[3] = 1;
    if(!i.equals(ai.get_index())) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(0,0,0,0) doesn't return (0,0,0,1).");
    }
    if(ai.get_abs_index() != 1) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(0,0,0,0) doesn't return 1.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void abs_index_test::test_inc_2() throw(libtest::test_exception) {

    static const char *testname = "abs_index_test::test_inc_2()";

    try {

    index<4> i1, i2;
    i2[0] = 1; i2[1] = 1; i2[2] = 1; i2[3] = 1;
    dimensions<4> dims(index_range<4>(i1, i2));
    index<4> i;
    i[0] = 1; i[1] = 1; i[2] = 0; i[3] = 0;
    abs_index<4> ai(i, dims);

    if(!ai.inc()) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,0,0) doesn't return true.");
    }
    i[0] = 1; i[1] = 1; i[2] = 0; i[3] = 1;
    if(!i.equals(ai.get_index())) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,0,0) doesn't return (1,1,0,1).");
    }
    if(ai.get_abs_index() != 13) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,0,0) doesn't return 13.");
    }
    if(!ai.inc()) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,0,1) doesn't return true.");
    }
    i[0] = 1; i[1] = 1; i[2] = 1; i[3] = 0;
    if(!i.equals(ai.get_index())) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,0,1) doesn't return (1,1,1,0).");
    }
    if(ai.get_abs_index() != 14) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,0,1) doesn't return 14.");
    }
    if(!ai.inc()) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,1,0) doesn't return true.");
    }
    i[0] = 1; i[1] = 1; i[2] = 1; i[3] = 1;
    if(!i.equals(ai.get_index())) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,1,0) doesn't return (1,1,1,1).");
    }
    if(ai.get_abs_index() != 15) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,1,0) doesn't return 15.");
    }
    if(ai.inc()) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,1,1) doesn't return false.");
    }
    if(!i.equals(ai.get_index())) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,1,1) doesn't preserve the index.");
    }
    if(ai.get_abs_index() != 15) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,1,1) doesn't preserve the absolute index.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void abs_index_test::test_inc_3() throw(libtest::test_exception) {

    static const char *testname = "abs_index_test::test_inc_3()";

    try {

    index<2> i1, i2;
    i2[0] = 10; i2[1] = 12;
    dimensions<2> dims(index_range<2>(i1, i2));
    index<2> i;
    i[0] = 0; i[1] = 11;
    abs_index<2> ai(i, dims);

    if(!ai.inc()) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(0,11) doesn't return true.");
    }
    i[0] = 0; i[1] = 12;
    if(!i.equals(ai.get_index())) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(0,11) doesn't return (0,12).");
    }
    if(ai.get_abs_index() != 12) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(0,11) doesn't return 12.");
    }
    if(!ai.inc()) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(0,12) doesn't return true.");
    }
    i[0] = 1; i[1] = 0;
    if(!i.equals(ai.get_index())) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(0,12) doesn't return (1,0).");
    }
    if(ai.get_abs_index() != 13) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(0,12) doesn't return 13.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void abs_index_test::test_inc_4() throw(libtest::test_exception) {

    static const char *testname = "abs_index_test::test_inc_4()";

    try {

    index<1> i1, i2;
    i2[0] = 5;
    dimensions<1> dims(index_range<1>(i1, i2));
    index<1> i;
    abs_index<1> ai(i, dims);

    if(!ai.inc()) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(0) doesn't return true.");
    }
    i[0] = 1;
    if(!i.equals(ai.get_index())) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(0) doesn't return (1).");
    }
    if(ai.get_abs_index() != 1) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(0) doesn't return 1.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void abs_index_test::test_inc_5() throw(libtest::test_exception) {

    static const char *testname = "abs_index_test::test_inc_5()";

    try {

    index<1> i1, i2;
    i2[0] = 5;
    dimensions<1> dims(index_range<1>(i1, i2));
    index<1> i;
    i[0] = 4;
    abs_index<1> ai(i, dims);

    if(!ai.inc()) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(4) doesn't return true.");
    }
    i[0] = 5;
    if(!i.equals(ai.get_index())) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(4) doesn't return (5).");
    }
    if(ai.get_abs_index() != 5) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(4) doesn't return 5.");
    }
    if(ai.inc()) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(5) doesn't return false.");
    }
    if(!i.equals(ai.get_index())) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(5) doesn't preserve the index.");
    }
    if(ai.get_abs_index() != 5) {
        fail_test(testname, __FILE__, __LINE__,
            "inc(5) doesn't preserve the absolute index.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void abs_index_test::test_last_1() throw(libtest::test_exception) {

    static const char *testname = "abs_index_test::test_last_1()";

    try {

    index<4> i1, i2;
    i2[0] = 1; i2[1] = 1; i2[2] = 1; i2[3] = 1;
    dimensions<4> dims(index_range<4>(i1, i2));

    i1[0] = 1; i1[1] = 1; i1[2] = 0; i1[3] = 0;
    abs_index<4> ii1(i1, dims);

    if(ii1.is_last()) {
        fail_test(testname, __FILE__, __LINE__,
            "[1,1,0,0] returns is_last() = true in [2,2,2,2]");
    }

    i1[0] = 1; i1[1] = 1; i1[2] = 1; i1[3] = 1;
    abs_index<4> ii2(i1, dims);

    if(!ii2.is_last()) {
        fail_test(testname, __FILE__, __LINE__,
            "[1,1,1,1] returns is_last() = false in [2,2,2,2]");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void abs_index_test::test_get_index_1() throw(libtest::test_exception) {

    static const char *testname = "abs_index_test::test_get_index_1()";

    try {

    index<2> i1, i2;
    i2[0] = 5; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    magic_dimensions<2> mdims(dims);

    index<2> i_ref;
    i_ref[0] = 2; i_ref[1] = 3;
    abs_index<2>::get_index(25, dims, i1);
    abs_index<2>::get_index(25, mdims, i2);

    if(!i1.equals(i_ref)) {
        fail_test(testname, __FILE__, __LINE__, "!i1.equals(i_ref)");
    }
    if(!i2.equals(i_ref)) {
        fail_test(testname, __FILE__, __LINE__, "!i2.equals(i_ref)");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
