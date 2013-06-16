#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/iface/iface.h>
#include "../compare_ref.h"
#include "direct_product_test.h"

namespace libtensor {


void direct_product_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);

    try {

        test_label_1();
        test_tt_1();
        test_tt_2();
        test_te_1();
        test_et_1();
        test_ee_1();
        test_ee_2();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void direct_product_test::test_label_1() throw(libtest::test_exception) {

    static const char *testname = "direct_product_test::test_label_1()";

    try {

    bispace<1> sp_a(5);
    bispace<2> sp_ab(sp_a|sp_a);
    btensor<2> t1(sp_ab), t2(sp_ab);
    letter a, b, c, d;

    if(!(t1(a|b) * t2(c|d)).get_core().contains(a)) {
        fail_test(testname, __FILE__, __LINE__,
            "Letter a is missing from the result label.");
    }
    if(!(t1(a|b) * t2(c|d)).get_core().contains(b)) {
        fail_test(testname, __FILE__, __LINE__,
            "Letter b is missing from the result label.");
    }
    if(!(t1(a|b) * t2(c|d)).get_core().contains(c)) {
        fail_test(testname, __FILE__, __LINE__,
            "Letter c is missing from the result label.");
    }
    if(!(t1(a|b) * t2(c|d)).get_core().contains(d)) {
        fail_test(testname, __FILE__, __LINE__,
            "Letter d is missing from the result label.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_product_test::test_tt_1() throw(libtest::test_exception) {

    static const char *testname = "direct_product_test::test_tt_1()";

    try {

    bispace<1> sp_a(5);
    bispace<2> sp_ab(sp_a|sp_a);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<2> t1(sp_ab), t2(sp_ab);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    contraction2<2, 2, 0> contr;
    btod_contract2<2, 2, 0> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d;
    t3(a|b|c|d) = t1(a|b) * t2(c|d);

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_product_test::test_tt_2() throw(libtest::test_exception) {

    static const char *testname = "direct_product_test::test_tt_2()";

    try {

    bispace<1> sp_a(5);
    bispace<2> sp_ab(sp_a|sp_a);
    bispace<3> sp_abc(sp_a|sp_a|sp_a);

    btensor<1> t1(sp_a);
    btensor<2> t2(sp_ab);
    btensor<3> t3(sp_abc), t3_ref(sp_abc);

    btod_random<1>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    contraction2<1, 2, 0> contr;
    btod_contract2<1, 2, 0> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c;
    t3(a|b|c) = t1(a) * t2(b|c);

    compare_ref<3>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_product_test::test_te_1() throw(libtest::test_exception) {

    static const char *testname = "direct_product_test::test_te_1()";

    try {

    bispace<1> sp_a(5);
    bispace<2> sp_ab(sp_a|sp_a);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<2> t1(sp_ab), t2a(sp_ab), t2b(sp_ab), t2(sp_ab);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2a);
    btod_random<2>().perform(t2b);
    t1.set_immutable();
    t2a.set_immutable();
    t2b.set_immutable();

    btod_add<2> add2(t2a);
    add2.add_op(t2b, -0.5);
    add2.perform(t2);
    contraction2<2, 2, 0> contr;
    btod_contract2<2, 2, 0> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d;
    t3(a|b|c|d) = t1(a|b) * (t2a(c|d) - 0.5*t2b(c|d));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_product_test::test_et_1() throw(libtest::test_exception) {

    static const char *testname = "direct_product_test::test_et_1()";

    try {

    bispace<1> sp_a(5);
    bispace<2> sp_ab(sp_a|sp_a);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<2> t1a(sp_ab), t1b(sp_ab), t1(sp_ab);
    btensor<2> t2(sp_ab);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<2>().perform(t1a);
    btod_random<2>().perform(t1b);
    btod_random<2>().perform(t2);
    t1a.set_immutable();
    t1b.set_immutable();
    t2.set_immutable();

    btod_add<2> add1(t1a);
    add1.add_op(t1b, 2.0);
    add1.perform(t1);
    contraction2<2, 2, 0> contr;
    btod_contract2<2, 2, 0> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d;
    t3(a|b|c|d) = (t1a(a|b) + 2.0*t1b(a|b)) * t2(c|d);

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_product_test::test_ee_1() throw(libtest::test_exception) {

    static const char *testname = "direct_product_test::test_ee_1()";

    try {

    bispace<1> sp_a(5);
    bispace<2> sp_ab(sp_a|sp_a);
    bispace<4> sp_abcd(sp_a|sp_a|sp_a|sp_a);

    btensor<2> t1a(sp_ab), t1b(sp_ab), t1(sp_ab);
    btensor<2> t2a(sp_ab), t2b(sp_ab), t2(sp_ab);
    btensor<4> t3(sp_abcd), t3_ref(sp_abcd);

    btod_random<2>().perform(t1a);
    btod_random<2>().perform(t1b);
    btod_random<2>().perform(t2a);
    btod_random<2>().perform(t2b);
    t1a.set_immutable();
    t1b.set_immutable();
    t2a.set_immutable();
    t2b.set_immutable();

    btod_add<2> add1(t1a, 1.5);
    add1.add_op(t1b, 2.0);
    add1.perform(t1);
    btod_add<2> add2(t2a);
    add2.add_op(t2b, -0.5);
    add2.perform(t2);
    contraction2<2, 2, 0> contr;
    btod_contract2<2, 2, 0> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d;
    t3(a|b|c|d) = (1.5*t1a(a|b) + 2.0*t1b(a|b)) * (t2a(c|d) - 0.5*t2b(c|d));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void direct_product_test::test_ee_2() throw(libtest::test_exception) {

    static const char *testname = "direct_product_test::test_ee_2()";

    try {

    bispace<1> sp_a(5);
    bispace<2> sp_ab(sp_a|sp_a);
    bispace<3> sp_abc(sp_a|sp_a|sp_a);

    btensor<1> t1a(sp_a), t1b(sp_a), t1(sp_a);
    btensor<2> t2a(sp_ab), t2b(sp_ab), t2(sp_ab);
    btensor<3> t3(sp_abc), t3_ref(sp_abc);

    btod_random<1>().perform(t1a);
    btod_random<1>().perform(t1b);
    btod_random<2>().perform(t2a);
    btod_random<2>().perform(t2b);
    t1a.set_immutable();
    t1b.set_immutable();
    t2a.set_immutable();
    t2b.set_immutable();

    btod_add<1> add1(t1a, 1.5);
    add1.add_op(t1b, 2.0);
    add1.perform(t1);
    btod_add<2> add2(t2a);
    add2.add_op(t2b, -0.5);
    add2.perform(t2);
    contraction2<1, 2, 0> contr;
    btod_contract2<1, 2, 0> op(contr, t1, t2);
    op.perform(t3_ref);

    letter a, b, c, d;
    t3(a|c|d) = (1.5*t1a(a) + 2.0*t1b(a)) * (t2a(c|d) - 0.5*t2b(c|d));

    compare_ref<3>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
