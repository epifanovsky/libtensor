#include <sstream>
#include <libtensor/core/abs_index.h>
#include "../test_utils.h"

using namespace libtensor;

int test_ctor_1() {

    static const char testname[] = "abs_index_test::test_ctor_1()";

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    index<2> i;
    i[0] = 0; i[1] = 0;
    abs_index<2> ai(i, dims);

    if(ai.get_abs_index() != 0) {
        return fail_test(testname, __FILE__, __LINE__,
            "abs(0,0) in (10,10) doesn't return 0.");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ctor_2() {

    static const char testname[] = "abs_index_test::test_ctor_2()";

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    index<2> i;
    i[0] = 1; i[1] = 0;
    abs_index<2> ai(i, dims);

    if(ai.get_abs_index() != 10) {
        return fail_test(testname, __FILE__, __LINE__,
            "abs(1,0) in (10,10) doesn't return 10.");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ctor_3() {

    static const char testname[] = "abs_index_test::test_ctor_3()";

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    index<2> i;
    i[0] = 9; i[1] = 9;
    abs_index<2> ai(i, dims);

    if(ai.get_abs_index() != 99) {
        return fail_test(testname, __FILE__, __LINE__,
            "abs(9,9) in (10,10) doesn't return 99.");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ctor_4() {

    static const char testname[] = "abs_index_test::test_ctor_4()";

    try {

    index<4> i1, i2;
    i2[0] = 1; i2[1] = 4; i2[2] = 1; i2[3] = 13;
    dimensions<4> dims(index_range<4>(i1, i2));
    index<4> i;
    i[0] = 1; i[1] = 0; i[2] = 1; i[3] = 0;
    abs_index<4> ai(154, dims);

    if(!i.equals(ai.get_index())) {
        return fail_test(testname, __FILE__, __LINE__,
            "abs(154) in (2,5,2,14) doesn't return (1,0,1,0).");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_ctor_5() {

    static const char testname[] = "abs_index_test::test_ctor_5()";

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
        return fail_test(testname, __FILE__, __LINE__, "out_of_bounds expected.");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_inc_1() {

    static const char testname[] = "abs_index_test::test_inc_1()";

    try {

    index<4> i1, i2;
    i2[0] = 1; i2[1] = 1; i2[2] = 1; i2[3] = 1;
    dimensions<4> dims(index_range<4>(i1, i2));
    index<4> i;
    abs_index<4> ai(i, dims);

    if(!ai.inc()) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(0,0,0,0) doesn't return true.");
    }
    i[0] = 0; i[1] = 0; i[2] = 0; i[3] = 1;
    if(!i.equals(ai.get_index())) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(0,0,0,0) doesn't return (0,0,0,1).");
    }
    if(ai.get_abs_index() != 1) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(0,0,0,0) doesn't return 1.");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_inc_2() {

    static const char testname[] = "abs_index_test::test_inc_2()";

    try {

    index<4> i1, i2;
    i2[0] = 1; i2[1] = 1; i2[2] = 1; i2[3] = 1;
    dimensions<4> dims(index_range<4>(i1, i2));
    index<4> i;
    i[0] = 1; i[1] = 1; i[2] = 0; i[3] = 0;
    abs_index<4> ai(i, dims);

    if(!ai.inc()) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,0,0) doesn't return true.");
    }
    i[0] = 1; i[1] = 1; i[2] = 0; i[3] = 1;
    if(!i.equals(ai.get_index())) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,0,0) doesn't return (1,1,0,1).");
    }
    if(ai.get_abs_index() != 13) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,0,0) doesn't return 13.");
    }
    if(!ai.inc()) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,0,1) doesn't return true.");
    }
    i[0] = 1; i[1] = 1; i[2] = 1; i[3] = 0;
    if(!i.equals(ai.get_index())) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,0,1) doesn't return (1,1,1,0).");
    }
    if(ai.get_abs_index() != 14) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,0,1) doesn't return 14.");
    }
    if(!ai.inc()) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,1,0) doesn't return true.");
    }
    i[0] = 1; i[1] = 1; i[2] = 1; i[3] = 1;
    if(!i.equals(ai.get_index())) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,1,0) doesn't return (1,1,1,1).");
    }
    if(ai.get_abs_index() != 15) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,1,0) doesn't return 15.");
    }
    if(ai.inc()) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,1,1) doesn't return false.");
    }
    if(!i.equals(ai.get_index())) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,1,1) doesn't preserve the index.");
    }
    if(ai.get_abs_index() != 15) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(1,1,1,1) doesn't preserve the absolute index.");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_inc_3() {

    static const char testname[] = "abs_index_test::test_inc_3()";

    try {

    index<2> i1, i2;
    i2[0] = 10; i2[1] = 12;
    dimensions<2> dims(index_range<2>(i1, i2));
    index<2> i;
    i[0] = 0; i[1] = 11;
    abs_index<2> ai(i, dims);

    if(!ai.inc()) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(0,11) doesn't return true.");
    }
    i[0] = 0; i[1] = 12;
    if(!i.equals(ai.get_index())) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(0,11) doesn't return (0,12).");
    }
    if(ai.get_abs_index() != 12) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(0,11) doesn't return 12.");
    }
    if(!ai.inc()) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(0,12) doesn't return true.");
    }
    i[0] = 1; i[1] = 0;
    if(!i.equals(ai.get_index())) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(0,12) doesn't return (1,0).");
    }
    if(ai.get_abs_index() != 13) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(0,12) doesn't return 13.");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_inc_4() {

    static const char testname[] = "abs_index_test::test_inc_4()";

    try {

    index<1> i1, i2;
    i2[0] = 5;
    dimensions<1> dims(index_range<1>(i1, i2));
    index<1> i;
    abs_index<1> ai(i, dims);

    if(!ai.inc()) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(0) doesn't return true.");
    }
    i[0] = 1;
    if(!i.equals(ai.get_index())) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(0) doesn't return (1).");
    }
    if(ai.get_abs_index() != 1) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(0) doesn't return 1.");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_inc_5() {

    static const char testname[] = "abs_index_test::test_inc_5()";

    try {

    index<1> i1, i2;
    i2[0] = 5;
    dimensions<1> dims(index_range<1>(i1, i2));
    index<1> i;
    i[0] = 4;
    abs_index<1> ai(i, dims);

    if(!ai.inc()) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(4) doesn't return true.");
    }
    i[0] = 5;
    if(!i.equals(ai.get_index())) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(4) doesn't return (5).");
    }
    if(ai.get_abs_index() != 5) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(4) doesn't return 5.");
    }
    if(ai.inc()) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(5) doesn't return false.");
    }
    if(!i.equals(ai.get_index())) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(5) doesn't preserve the index.");
    }
    if(ai.get_abs_index() != 5) {
        return fail_test(testname, __FILE__, __LINE__,
            "inc(5) doesn't preserve the absolute index.");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_last_1() {

    static const char testname[] = "abs_index_test::test_last_1()";

    try {

    index<4> i1, i2;
    i2[0] = 1; i2[1] = 1; i2[2] = 1; i2[3] = 1;
    dimensions<4> dims(index_range<4>(i1, i2));

    i1[0] = 1; i1[1] = 1; i1[2] = 0; i1[3] = 0;
    abs_index<4> ii1(i1, dims);

    if(ii1.is_last()) {
        return fail_test(testname, __FILE__, __LINE__,
            "[1,1,0,0] returns is_last() = true in [2,2,2,2]");
    }

    i1[0] = 1; i1[1] = 1; i1[2] = 1; i1[3] = 1;
    abs_index<4> ii2(i1, dims);

    if(!ii2.is_last()) {
        return fail_test(testname, __FILE__, __LINE__,
            "[1,1,1,1] returns is_last() = false in [2,2,2,2]");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_get_index_1() {

    static const char testname[] = "abs_index_test::test_get_index_1()";

    try {

    index<2> i1, i2;
    i2[0] = 5; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    magic_dimensions<2> mdims(dims, true);

    index<2> i_ref;
    i_ref[0] = 2; i_ref[1] = 3;
    abs_index<2>::get_index(25, dims, i1);
    abs_index<2>::get_index(25, mdims, i2);

    if(!i1.equals(i_ref)) {
        return fail_test(testname, __FILE__, __LINE__, "!i1.equals(i_ref)");
    }
    if(!i2.equals(i_ref)) {
        return fail_test(testname, __FILE__, __LINE__, "!i2.equals(i_ref)");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_get_index_2() {

    static const char testname[] = "abs_index_test::test_get_index_2()";

    try {

        index<2> i1, i2;
        i2[0] = 5; i2[1] = 9;
        dimensions<2> dims1(index_range<2>(i1, i2));
        i2[0] = 9; i2[1] = 5;
        dimensions<2> dims2(index_range<2>(i1, i2));

        index<2> i, i_ref;

        magic_dimensions<2> mdims1(dims1, true);
        if(!mdims1.get_dims().equals(dims1)) {
            return fail_test(testname, __FILE__, __LINE__, "Bad dimensions (1)");
        }
        abs_index<2>::get_index(34, mdims1, i);
        i_ref[0] = 3; i_ref[1] = 4;
        if(!i.equals(i_ref)) {
            std::ostringstream ss;
            ss << "Bad index (1): " << i << " vs. " << i_ref << " (ref)";
            return fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }

        magic_dimensions<2> mdims2(mdims1);
        if(!mdims2.get_dims().equals(dims1)) {
            return fail_test(testname, __FILE__, __LINE__, "Bad dimensions (2)");
        }
        abs_index<2>::get_index(34, mdims2, i);
        i_ref[0] = 3; i_ref[1] = 4;
        if(!i.equals(i_ref)) {
            std::ostringstream ss;
            ss << "Bad index (2): " << i << " vs. " << i_ref << " (ref)";
            return fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }

        mdims1.permute(permutation<2>().permute(0, 1));
        if(!mdims1.get_dims().equals(dims2)) {
            return fail_test(testname, __FILE__, __LINE__, "Bad dimensions (3)");
        }
        abs_index<2>::get_index(34, mdims1, i);
        i_ref[0] = 5; i_ref[1] = 4;
        if(!i.equals(i_ref)) {
            std::ostringstream ss;
            ss << "Bad index (3): " << i << " vs. " << i_ref << " (ref)";
            return fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }

        mdims2.permute(permutation<2>());
        if(!mdims2.get_dims().equals(dims1)) {
            return fail_test(testname, __FILE__, __LINE__, "Bad dimensions (4)");
        }
        abs_index<2>::get_index(34, mdims2, i);
        i_ref[0] = 3; i_ref[1] = 4;
        if(!i.equals(i_ref)) {
            std::ostringstream ss;
            ss << "Bad index (4): " << i << " vs. " << i_ref << " (ref)";
            return fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    return

    test_ctor_1() |
    test_ctor_2() |
    test_ctor_3() |
    test_ctor_4() |
    test_ctor_5() |
    test_inc_1() |
    test_inc_2() |
    test_inc_3() |
    test_inc_4() |
    test_last_1() |
    test_get_index_1() |
    test_get_index_2() |

    0;
}

