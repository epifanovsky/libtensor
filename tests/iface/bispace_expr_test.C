#include <sstream>
#include <libtensor/iface/bispace.h>
#include "bispace_expr_test.h"

namespace libtensor {


void bispace_expr_test::perform() throw(libtest::test_exception) {

    test_sym_1();
    test_sym_2();
    test_sym_3();
    test_sym_4();
    test_sym_5();
    test_sym_6();
    test_sym_7();
    test_sym_8();
    test_sym_9();
    test_sym_10();

    test_contains_1();
    test_contains_2();
    test_contains_3();
    test_contains_4();

    test_locate_1();
    test_locate_2();
    test_locate_3();
    test_locate_4();

    test_perm_1();
    test_perm_2();
    test_perm_3();
    test_perm_4();
    test_perm_5();
    test_perm_6();

    test_exc_1();
}


void bispace_expr_test::test_sym_1() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_sym_1()";

    try {

    bispace<1> a(10), b(10);
    mask<2> msk, msk_ref;
    msk_ref[0] = true; msk_ref[1] = true;
    (a&b).mark_sym(0, msk);
    if(!msk.equals(msk_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask: " << msk << " vs. " << msk_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_sym_2() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_sym_2()";

    try {

    bispace<1> a(10), b(10), c(10);
    mask<3> msk1, msk2, msk3, msk_ref;
    msk_ref[0] = true; msk_ref[1] = true; msk_ref[2] = true;
    (a&b&c).mark_sym(0, msk1);
    (a&b&c).mark_sym(1, msk2);
    (a&b&c).mark_sym(3, msk3);
    if(!msk1.equals(msk_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 1: " << msk1 << " vs. " << msk_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk2.equals(msk_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 2: " << msk2 << " vs. " << msk_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk3.equals(msk_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 3: " << msk3 << " vs. " << msk_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_sym_3() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_sym_3()";

    try {

    bispace<1> a(10), b(10);
    mask<2> msk1, msk1_ref;
    mask<2> msk2, msk2_ref;
    msk1_ref[0] = true;
    msk2_ref[1] = true;
    (a|b).mark_sym(0, msk1);
    (a|b).mark_sym(1, msk2);
    if(!msk1.equals(msk1_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 1: " << msk1 << " vs. " << msk1_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk2.equals(msk2_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 2: " << msk2 << " vs. " << msk2_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_sym_4() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_sym_4()";

    try {

    bispace<1> a(10), b(10), c(10), d(10);
    mask<4> msk1, msk1_ref;
    mask<4> msk2, msk2_ref;
    mask<4> msk3, msk3_ref;
    mask<4> msk4, msk4_ref;
    msk1_ref[0] = true;
    msk2_ref[1] = true;
    msk3_ref[2] = true;
    msk4_ref[3] = true;
    ((a|b)|(c|d)).mark_sym(0, msk1);
    ((a|b)|(c|d)).mark_sym(1, msk2);
    ((a|b)|(c|d)).mark_sym(2, msk3);
    ((a|b)|(c|d)).mark_sym(3, msk4);
    if(!msk1.equals(msk1_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 1: " << msk1 << " vs. " << msk1_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk2.equals(msk2_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 2: " << msk2 << " vs. " << msk2_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk3.equals(msk3_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 3: " << msk3 << " vs. " << msk3_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk4.equals(msk4_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 4: " << msk4 << " vs. " << msk4_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_sym_5() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_sym_5()";

    try {

    bispace<1> a(10), b(10), c(10), d(10);
    mask<4> msk1, msk1_ref;
    mask<4> msk2, msk2_ref;
    mask<4> msk3, msk3_ref;
    mask<4> msk4, msk4_ref;
    msk1_ref[0] = true; msk1_ref[1] = true;
    msk2_ref[0] = true; msk2_ref[1] = true;
    msk3_ref[2] = true; msk3_ref[3] = true;
    msk4_ref[2] = true; msk4_ref[3] = true;
    ((a&b)|(c&d)).mark_sym(0, msk1);
    ((a&b)|(c&d)).mark_sym(1, msk2);
    ((a&b)|(c&d)).mark_sym(2, msk3);
    ((a&b)|(c&d)).mark_sym(3, msk4);
    if(!msk1.equals(msk1_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 1: " << msk1 << " vs. " << msk1_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk2.equals(msk2_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 2: " << msk2 << " vs. " << msk2_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk3.equals(msk3_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 3: " << msk3 << " vs. " << msk3_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk4.equals(msk4_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 4: " << msk4 << " vs. " << msk4_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_sym_6() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_sym_6()";

    try {

    bispace<1> a(10), b(10), c(10), d(10);
    mask<4> msk1, msk1_ref;
    mask<4> msk2, msk2_ref;
    mask<4> msk3, msk3_ref;
    mask<4> msk4, msk4_ref;
    msk1_ref[0] = true; msk1_ref[2] = true;
    msk2_ref[1] = true; msk2_ref[3] = true;
    msk3_ref[0] = true; msk3_ref[2] = true;
    msk4_ref[1] = true; msk4_ref[3] = true;
    ((a|b)&(c|d)).mark_sym(0, msk1);
    ((a|b)&(c|d)).mark_sym(1, msk2);
    ((a|b)&(c|d)).mark_sym(2, msk3);
    ((a|b)&(c|d)).mark_sym(3, msk4);
    if(!msk1.equals(msk1_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 1: " << msk1 << " vs. " << msk1_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk2.equals(msk2_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 2: " << msk2 << " vs. " << msk2_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk3.equals(msk3_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 3: " << msk3 << " vs. " << msk3_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk4.equals(msk4_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 4: " << msk4 << " vs. " << msk4_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_sym_7() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_sym_7()";

    try {

    bispace<1> a(10), b(10), c(10), d(10), e(10), f(10);
    mask<6> msk1, msk1_ref;
    mask<6> msk2, msk2_ref;
    mask<6> msk3, msk3_ref;
    mask<6> msk4, msk4_ref;
    mask<6> msk5, msk5_ref;
    mask<6> msk6, msk6_ref;
    msk1_ref[0] = true; msk1_ref[2] = true; msk1_ref[4] = true;
    msk2_ref[1] = true; msk2_ref[3] = true; msk2_ref[5] = true;
    msk3_ref[0] = true; msk3_ref[2] = true; msk3_ref[4] = true;
    msk4_ref[1] = true; msk4_ref[3] = true; msk4_ref[5] = true;
    msk5_ref[0] = true; msk5_ref[2] = true; msk5_ref[4] = true;
    msk6_ref[1] = true; msk6_ref[3] = true; msk6_ref[5] = true;
    ((a|b)&(c|d)&(e|f)).mark_sym(0, msk1);
    ((a|b)&(c|d)&(e|f)).mark_sym(1, msk2);
    ((a|b)&(c|d)&(e|f)).mark_sym(2, msk3);
    ((a|b)&(c|d)&(e|f)).mark_sym(3, msk4);
    ((a|b)&(c|d)&(e|f)).mark_sym(4, msk5);
    ((a|b)&(c|d)&(e|f)).mark_sym(5, msk6);
    if(!msk1.equals(msk1_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 1: " << msk1 << " vs. " << msk1_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk2.equals(msk2_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 2: " << msk2 << " vs. " << msk2_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk3.equals(msk3_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 3: " << msk3 << " vs. " << msk3_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk4.equals(msk4_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 4: " << msk4 << " vs. " << msk4_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk5.equals(msk5_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 5: " << msk5 << " vs. " << msk5_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk6.equals(msk6_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 6: " << msk6 << " vs. " << msk6_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_sym_8() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_sym_8()";

    try {

    bispace<1> a(10), b(10), c(10), d(10), e(10), f(10);
    mask<6> msk1, msk1_ref;
    mask<6> msk2, msk2_ref;
    mask<6> msk3, msk3_ref;
    mask<6> msk4, msk4_ref;
    mask<6> msk5, msk5_ref;
    mask<6> msk6, msk6_ref;
    msk1_ref[0] = true; msk1_ref[1] = true; msk1_ref[2] = true;
    msk2_ref[0] = true; msk2_ref[1] = true; msk2_ref[2] = true;
    msk3_ref[0] = true; msk3_ref[1] = true; msk3_ref[2] = true;
    msk4_ref[3] = true; msk4_ref[4] = true; msk4_ref[5] = true;
    msk5_ref[3] = true; msk5_ref[4] = true; msk5_ref[5] = true;
    msk6_ref[3] = true; msk6_ref[4] = true; msk6_ref[5] = true;
    ((a&b&c)|(d&e&f)).mark_sym(0, msk1);
    ((a&b&c)|(d&e&f)).mark_sym(1, msk2);
    ((a&b&c)|(d&e&f)).mark_sym(2, msk3);
    ((a&b&c)|(d&e&f)).mark_sym(3, msk4);
    ((a&b&c)|(d&e&f)).mark_sym(4, msk5);
    ((a&b&c)|(d&e&f)).mark_sym(5, msk6);
    if(!msk1.equals(msk1_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 1: " << msk1 << " vs. " << msk1_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk2.equals(msk2_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 2: " << msk2 << " vs. " << msk2_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk3.equals(msk3_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 3: " << msk3 << " vs. " << msk3_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk4.equals(msk4_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 4: " << msk4 << " vs. " << msk4_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk5.equals(msk5_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 5: " << msk5 << " vs. " << msk5_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk6.equals(msk6_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 6: " << msk6 << " vs. " << msk6_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_sym_9() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_sym_9()";

    try {

    bispace<1> a(10), b(10);
    bispace<2> ab(a&b), cd(a&b);
    mask<4> msk1, msk1_ref;
    mask<4> msk2, msk2_ref;
    mask<4> msk3, msk3_ref;
    mask<4> msk4, msk4_ref;
    msk1_ref[0] = true; msk1_ref[1] = true;
    msk2_ref[0] = true; msk2_ref[1] = true;
    msk3_ref[2] = true; msk3_ref[3] = true;
    msk4_ref[2] = true; msk4_ref[3] = true;
    (ab|cd).mark_sym(0, msk1);
    (ab|cd).mark_sym(1, msk2);
    (ab|cd).mark_sym(2, msk3);
    (ab|cd).mark_sym(3, msk4);
    if(!msk1.equals(msk1_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 1: " << msk1 << " vs. " << msk1_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk2.equals(msk2_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 2: " << msk2 << " vs. " << msk2_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk3.equals(msk3_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 3: " << msk3 << " vs. " << msk3_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!msk4.equals(msk4_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 4: " << msk4 << " vs. " << msk4_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_sym_10() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_sym_10()";

    try {

    bispace<1> a(10);
    mask<4> m1, m2, m3, m4, msk_ref;
    msk_ref[0] = true; msk_ref[1] = true;
    msk_ref[2] = true; msk_ref[3] = true;
    (a&a&a&a).mark_sym(0, m1);
    (a&a&a&a).mark_sym(1, m2);
    (a&a&a&a).mark_sym(2, m3);
    (a&a&a&a).mark_sym(3, m4);
    if(!m1.equals(msk_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 1: " << m1 << " vs. " << msk_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!m2.equals(msk_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 2: " << m2 << " vs. " << msk_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!m3.equals(msk_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 3: " << m3 << " vs. " << msk_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(!m4.equals(msk_ref)) {
        std::ostringstream ss;
        ss << "Unexpected mask 4: " << m4 << " vs. " << msk_ref
            << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_contains_1() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_contains_1()";

    try {

    bispace<1> a(10), b(10), c(10);

    size_t conta = (a|b).contains(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(a));
    size_t contb = (a|b).contains(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(b));
    size_t contc = (a|b).contains(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(c));

    if(conta != 1) {
        fail_test(testname, __FILE__, __LINE__,
            "contains(a).");
    }
    if(contb != 1) {
        fail_test(testname, __FILE__, __LINE__,
            "contains(b).");
    }
    if(contc != 0) {
        fail_test(testname, __FILE__, __LINE__,
            "contains(c).");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_contains_2() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_contains_2()";

    try {

    bispace<1> a(10), b(10), c(10);

    size_t conta = (a&b).contains(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(a));
    size_t contb = (a&b).contains(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(b));
    size_t contc = (a&b).contains(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(c));

    if(conta != 1) {
        fail_test(testname, __FILE__, __LINE__,
            "contains(a).");
    }
    if(contb != 1) {
        fail_test(testname, __FILE__, __LINE__,
            "contains(b).");
    }
    if(contc != 0) {
        fail_test(testname, __FILE__, __LINE__,
            "contains(c).");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_contains_3() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_contains_3()";

    try {

    bispace<1> a(10), b(10), c(10), d(10), e(10);

    size_t conta = ((a&b)|(c&d)).contains(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(a));
    size_t contb = ((a&b)|(c&d)).contains(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(b));
    size_t contc = ((a&b)|(c&d)).contains(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(c));
    size_t contd = ((a&b)|(c&d)).contains(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(d));
    size_t conte = ((a&b)|(c&d)).contains(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(e));

    if(conta != 1) {
        fail_test(testname, __FILE__, __LINE__,
            "contains(a).");
    }
    if(contb != 1) {
        fail_test(testname, __FILE__, __LINE__,
            "contains(b).");
    }
    if(contc != 1) {
        fail_test(testname, __FILE__, __LINE__,
            "contains(c).");
    }
    if(contd != 1) {
        fail_test(testname, __FILE__, __LINE__,
            "contains(d).");
    }
    if(conte != 0) {
        fail_test(testname, __FILE__, __LINE__,
            "contains(e).");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_contains_4() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_contains_4()";

    try {

    bispace<1> a(10), b(10);
    bispace<2> ab(a&b), cd(a&b);

    size_t conta = (ab|cd).contains(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(a));
    size_t contb = (ab|cd).contains(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(b));
    size_t contab = (ab|cd).contains(
        bispace_expr::expr< 2, bispace_expr::ident<2> >(ab));
    size_t contcd = (ab|cd).contains(
        bispace_expr::expr< 2, bispace_expr::ident<2> >(cd));

    if(conta != 0) {
        fail_test(testname, __FILE__, __LINE__,
            "contains(a).");
    }
    if(contb != 0) {
        fail_test(testname, __FILE__, __LINE__,
            "contains(b).");
    }
    if(contab != 1) {
        fail_test(testname, __FILE__, __LINE__,
            "contains(ab).");
    }
    if(contcd != 1) {
        fail_test(testname, __FILE__, __LINE__,
            "contains(cd).");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_locate_1() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_locate_1()";

    try {

    bispace<1> a(10), b(10);

    size_t loca = (a|b).locate(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(a));
    size_t locb = (a|b).locate(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(b));

    if(loca != 0) {
        std::ostringstream ss;
        ss << "locate(a) = " << loca << " (expected 0).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(locb != 1) {
        std::ostringstream ss;
        ss << "locate(b) = " << loca << " (expected 1).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_locate_2() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_locate_2()";

    try {

    bispace<1> a(10), b(10);

    size_t loca = (a&b).locate(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(a));
    size_t locb = (a&b).locate(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(b));

    if(loca != 0) {
        std::ostringstream ss;
        ss << "locate(a) = " << loca << " (expected 0).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(locb != 1) {
        std::ostringstream ss;
        ss << "locate(b) = " << loca << " (expected 1).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_locate_3() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_locate_3()";

    try {

    bispace<1> a(10), b(10), c(10), d(10), e(10);

    size_t loca = (a&b|c&d|e).locate(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(a));
    size_t locb = (a&b|c&d|e).locate(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(b));
    size_t locc = (a&b|c&d|e).locate(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(c));
    size_t locd = (a&b|c&d|e).locate(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(d));
    size_t loce = (a&b|c&d|e).locate(
        bispace_expr::expr< 1, bispace_expr::ident<1> >(e));

    if(loca != 0) {
        std::ostringstream ss;
        ss << "locate(a) = " << loca << " (expected 0).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(locb != 1) {
        std::ostringstream ss;
        ss << "locate(b) = " << loca << " (expected 1).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(locc != 2) {
        std::ostringstream ss;
        ss << "locate(c) = " << locc << " (expected 2).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(locd != 3) {
        std::ostringstream ss;
        ss << "locate(d) = " << locd << " (expected 3).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(loce != 4) {
        std::ostringstream ss;
        ss << "locate(e) = " << loce << " (expected 4).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_locate_4() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_locate_4()";

    try {

    bispace<1> a(10), b(10);
    bispace<2> ab(a&b), cd(a&b);

    size_t locab = (ab|cd).locate(
        bispace_expr::expr< 2, bispace_expr::ident<2> >(ab));
    size_t loccd = (ab|cd).locate(
        bispace_expr::expr< 2, bispace_expr::ident<2> >(cd));

    if(locab != 0) {
        std::ostringstream ss;
        ss << "locate(ab) = " << locab << " (expected 0).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(loccd != 2) {
        std::ostringstream ss;
        ss << "locate(cd) = " << loccd << " (expected 2).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_perm_1() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_perm_1()";

    try {

    bispace<1> a(10), b(10);
    permutation<2> perm, perm_ref;
    (a|b).build_permutation(a|b, perm);
    if(!perm.equals(perm_ref)) {
        std::ostringstream ss;
        ss << "Unexpected permutation for (a|b)<-(a|b): "
            << perm << " vs. " << perm_ref << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_perm_2() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_perm_2()";

    try {

    bispace<1> a(10), b(10);
    permutation<2> perm, perm_ref;
    perm_ref.permute(0, 1);
    (a|b).build_permutation(b|a, perm);
    if(!perm.equals(perm_ref)) {
        std::ostringstream ss;
        ss << "Unexpected permutation for (a|b)<-(b|a): "
            << perm << " vs. " << perm_ref << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_perm_3() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_perm_3()";

    try {

    bispace<1> a(10), b(10), c(10), d(10);
    permutation<4> perm, perm_ref;
    perm_ref.permute(2, 3).permute(1, 2).permute(0, 1);
    (a|b|c|d).build_permutation(b|c|d|a, perm);
    if(!perm.equals(perm_ref)) {
        std::ostringstream ss;
        ss << "Unexpected permutation for (a|b|c|d)<-(b|c|d|a): "
            << perm << " vs. " << perm_ref << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_perm_4() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_perm_4()";

    try {

    bispace<1> a(10), b(10), c(10), d(10);
    bispace<2> ab(a&b), cd(a&b);
    permutation<4> perm, perm_ref;
    (ab|cd).build_permutation(ab|cd, perm);
    if(!perm.equals(perm_ref)) {
        std::ostringstream ss;
        ss << "Unexpected permutation for (ab|cd)<-(ab|cd): "
            << perm << " vs. " << perm_ref << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_perm_5() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_perm_5()";

    try {

    bispace<1> a(10), b(10), c(10), d(10);
    bispace<2> ab(a&b), cd(a&b);
    permutation<4> perm, perm_ref;
    perm_ref.permute(0, 2).permute(1, 3);
    (ab|cd).build_permutation(cd|ab, perm);
    if(!perm.equals(perm_ref)) {
        std::ostringstream ss;
        ss << "Unexpected permutation for (ab|cd)<-(cd|ab): "
            << perm << " vs. " << perm_ref << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_perm_6() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_perm_6()";

    try {

    bispace<1> a(10), b(10), c(10), d(10), e(20);
    bispace<2> ab(a&b), cd(a&b);
    permutation<5> perm, perm_ref;
    perm_ref.permute(0, 1).permute(1, 2).permute(0, 3).permute(1, 4);
    (ab|e|cd).build_permutation(e|cd&ab, perm);
    if(!perm.equals(perm_ref)) {
        std::ostringstream ss;
        ss << "Unexpected permutation for (ab|e|cd)<-(e|cd&ab): "
            << perm << " vs. " << perm_ref << " (ref).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void bispace_expr_test::test_exc_1() throw(libtest::test_exception) {

    static const char *testname = "bispace_expr_test::test_exc_1()";

    try {

    bispace<1> a(10), b(20);

    bool ok = false;
    try {
        (a&b);
    } catch(expr_exception &e) {
        ok = true;
    }
    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "Exception expected with incompatible bispaces.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
