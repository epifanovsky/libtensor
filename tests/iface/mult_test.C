#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/btod_add.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_mult.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/block_tensor/btod_set.h>
#include <libtensor/libtensor.h>
#include "../compare_ref.h"
#include "mult_test.h"

namespace libtensor {


void mult_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);

    try {
        test_tt_1a();
        test_tt_1b();
        test_tt_2();
        test_tt_3();
        test_tt_4();
        test_tt_5();
        test_tt_6a();
        test_tt_6b();
        test_te_1();
        test_te_2();
        test_te_3();
        test_et_1();
        test_et_2();
        test_et_3();
        test_ee_1a();
        test_ee_1b();
        test_ee_2();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void mult_test::test_tt_1a() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_tt_1a()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);

    btensor<2> t1(sp_ia), t2(sp_ia), t3(sp_ia), t3_ref(sp_ia);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    btod_mult<2>(t1, t2, false).perform(t3_ref);

    letter i, a;
    t3(i|a) = mult(t1(i|a), t2(i|a));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void mult_test::test_tt_1b() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_tt_1b()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);

    btensor<2> t1(sp_ia), t2(sp_ia), t3(sp_ia), t3_ref(sp_ia);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    btod_mult<2>(t1, t2, true).perform(t3_ref);

    letter i, a;
    t3(i|a) = div(t1(i|a), t2(i|a));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void mult_test::test_tt_2() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_tt_2()";

    try {

    bispace<1> sp_i(10);
    bispace<3> sp_ijk(sp_i|sp_i|sp_i);

    btensor<3> t1(sp_ijk), t2(sp_ijk), t3(sp_ijk), t3_ref(sp_ijk);

    btod_random<3>().perform(t1);
    btod_random<3>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    permutation<3> perm1, perm2;
    perm2.permute(0, 1).permute(1, 2);

    btod_mult<3>(t1, perm1, t2, perm2, false).perform(t3_ref);

    letter i, j, k;
    t3(i|j|k) = mult(t1(i|j|k), t2(k|i|j));

    compare_ref<3>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void mult_test::test_tt_3() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_tt_3()";

    try {

    bispace<1> sp_i(10);
    bispace<3> sp_ijk(sp_i|sp_i|sp_i);

    btensor<3> t1(sp_ijk), t2(sp_ijk), t3(sp_ijk), t3_ref(sp_ijk);

    btod_random<3>().perform(t1);
    btod_random<3>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    permutation<3> perm1, perm2;
    perm1.permute(1, 2).permute(0, 1);

    btod_mult<3>(t1, perm1, t2, perm2, false).perform(t3_ref);

    letter i, j, k;
    t3(i|j|k) = mult(t1(j|k|i), t2(i|j|k));

    compare_ref<3>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void mult_test::test_tt_4() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_tt_4()";

    try {

    bispace<1> sp_i(10);
    bispace<3> sp_ijk(sp_i&sp_i&sp_i);

    btensor<3> t1(sp_ijk), t2(sp_ijk), t3(sp_ijk), t3_ref(sp_ijk);

    btod_random<3>().perform(t1);
    btod_random<3>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    permutation<3> perm1, perm2;
    perm1.permute(1, 2).permute(0, 1);
    perm2.permute(0, 1).permute(1, 2);

    btod_mult<3>(t1, perm1, t2, perm2, false).perform(t3_ref);

    letter i, j, k;
    t3(i|j|k) = mult(t1(j|k|i), t2(k|i|j));

    compare_ref<3>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void mult_test::test_tt_5() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_tt_5()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);

    btensor<2> t1(sp_ia), t2(sp_ia), t3(sp_ia), t3_ref(sp_ia);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    btod_mult<2>(t1, t2, false, -0.3).perform(t3_ref);

    letter i, a;
    t3(i|a) = 0.2 * mult(-1.5 * t1(i|a), t2(i|a));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void mult_test::test_tt_6a() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_tt_6a()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);

    btensor<2> t1(sp_ia), t2(sp_ia), t3(sp_ia), t3_ref(sp_ia);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    btod_mult<2>(t1, t2, false, 0.2).perform(t3_ref);

    letter i, a;
    t3(i|a) = 0.1 * mult(t1(i|a), 2.0 * t2(i|a));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void mult_test::test_tt_6b() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_tt_6b()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);

    btensor<2> t1(sp_ia), t2(sp_ia), t3(sp_ia), t3_ref(sp_ia);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    btod_mult<2>(t1, t2, true, 0.8).perform(t3_ref);

    letter i, a;
    t3(i|a) = 1.6 * div(t1(i|a), 2.0 * t2(i|a));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void mult_test::test_te_1() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_te_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);

    btensor<2> t1(sp_ia);
    btensor<2> t21(sp_ia), t22(sp_ia), t2(sp_ia);
    btensor<2> t3(sp_ia), t3_ref(sp_ia);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t21);
    btod_random<2>().perform(t22);
    t1.set_immutable();
    t21.set_immutable();
    t22.set_immutable();

    btod_add<2> add2(t21, 1.0);
    add2.add_op(t22, 1.0);
    add2.perform(t2);
    t2.set_immutable();

    btod_set<2>(1.0).perform(t3);
    btod_set<2>(1.0).perform(t3_ref);
    //~ btod_copy<2>(t3_ref, -1.0).perform(t3, 1.0);
    //~ btod_copy<2>(t3).perform(t3_ref);

    btod_mult<2>(t1, t2, false).perform(t3_ref);

    letter i, a;
    t3(i|a) = mult(t1(i|a), t21(i|a) + t22(i|a));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void mult_test::test_te_2() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_te_2()";

    try {

    bispace<1> sp_i(10);
    bispace<3> sp_ijk(sp_i&sp_i&sp_i);

    btensor<3> t1(sp_ijk), t21(sp_ijk), t22(sp_ijk), t2(sp_ijk);
    btensor<3> t3(sp_ijk), t3_ref(sp_ijk);

    btod_random<3>().perform(t1);
    btod_random<3>().perform(t21);
    btod_random<3>().perform(t22);
    t1.set_immutable();
    t21.set_immutable();
    t22.set_immutable();
    btod_add<3> add2(t21, 1.0);
    add2.add_op(t22, 1.0);
    add2.perform(t2);
    t2.set_immutable();

    permutation<3> perm1, perm2;
    perm2.permute(0, 1).permute(1, 2);

    btod_mult<3>(t1, perm1, t2, perm2, false).perform(t3_ref);

    letter i, j, k;
    t3(i|j|k) = mult(t1(i|j|k), t21(k|i|j) + t22(k|i|j));

    compare_ref<3>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void mult_test::test_te_3() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_te_3()";

    try {

    bispace<1> sp_i(10);
    bispace<3> sp_ijk(sp_i&sp_i&sp_i);

    btensor<3> t1(sp_ijk), t21(sp_ijk), t22(sp_ijk), t2(sp_ijk);
    btensor<3> t3(sp_ijk), t3_ref(sp_ijk);

    btod_random<3>().perform(t1);
    btod_random<3>().perform(t21);
    btod_random<3>().perform(t22);
    t1.set_immutable();
    t21.set_immutable();
    t22.set_immutable();
    btod_add<3> add2(t21, 1.0);
    add2.add_op(t22, 1.0);
    add2.perform(t2);
    t2.set_immutable();

    permutation<3> perm1, perm2;
    perm1.permute(1, 2).permute(0, 1);

    btod_mult<3>(t1, perm1, t2, perm2, false).perform(t3_ref);

    letter i, j, k;
    t3(i|j|k) = mult(t1(j|k|i), t21(i|j|k) + t22(i|j|k));

    compare_ref<3>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void mult_test::test_et_1() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_et_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);

    btensor<2> t11(sp_ia), t12(sp_ia), t1(sp_ia);
    btensor<2> t2(sp_ia);
    btensor<2> t3(sp_ia), t3_ref(sp_ia);

    btod_random<2>().perform(t11);
    btod_random<2>().perform(t12);
    btod_random<2>().perform(t2);
    t11.set_immutable();
    t12.set_immutable();
    t2.set_immutable();

    btod_add<2> add1(t11, 1.0);
    add1.add_op(t12, 1.0);
    add1.perform(t1);
    t1.set_immutable();

    btod_mult<2>(t1, t2, false).perform(t3_ref);

    letter i, a;
    t3(i|a) = mult(t11(i|a) + t12(i|a), t2(i|a));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void mult_test::test_et_2() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_et_2()";

    try {

    bispace<1> sp_i(10);
    bispace<3> sp_ijk(sp_i&sp_i&sp_i);

    btensor<3> t11(sp_ijk), t12(sp_ijk), t1(sp_ijk), t2(sp_ijk);
    btensor<3> t3(sp_ijk), t3_ref(sp_ijk);

    btod_random<3>().perform(t11);
    btod_random<3>().perform(t12);
    btod_random<3>().perform(t2);
    t11.set_immutable();
    t12.set_immutable();
    t2.set_immutable();
    btod_add<3> add1(t11, 1.0);
    add1.add_op(t12, 1.0);
    add1.perform(t1);
    t1.set_immutable();

    permutation<3> perm1, perm2;
    perm2.permute(1, 2).permute(0, 1);

    btod_mult<3>(t1, perm1, t2, perm2, false).perform(t3_ref);

    letter i, j, k;
    t3(i|j|k) = mult(t11(i|j|k) + t12(i|j|k), t2(j|k|i));

    compare_ref<3>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void mult_test::test_et_3() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_et_3()";

    try {

    bispace<1> sp_i(10);
    bispace<3> sp_ijk(sp_i&sp_i&sp_i);

    btensor<3> t11(sp_ijk), t12(sp_ijk), t1(sp_ijk), t2(sp_ijk);
    btensor<3> t3(sp_ijk), t3_ref(sp_ijk);

    btod_random<3>().perform(t11);
    btod_random<3>().perform(t12);
    btod_random<3>().perform(t2);
    t11.set_immutable();
    t12.set_immutable();
    t2.set_immutable();

    btod_add<3> add1(t11, 1.0);
    add1.add_op(t12, 1.0);
    add1.perform(t1);
    t1.set_immutable();

    permutation<3> perm1, perm2;
    perm1.permute(1, 2).permute(0, 1);

    btod_mult<3>(t1, perm1, t2, perm2, false).perform(t3_ref);

    letter i, j, k;
    t3(i|j|k) = mult(t11(j|k|i) + t12(j|k|i), t2(i|j|k));

    compare_ref<3>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void mult_test::test_ee_1a() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_ee_1a()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);

    btensor<2> t11(sp_ia), t12(sp_ia), t1(sp_ia);
    btensor<2> t21(sp_ia), t22(sp_ia), t2(sp_ia);
    btensor<2> t3(sp_ia), t3_ref(sp_ia);

    btod_random<2>().perform(t11);
    btod_random<2>().perform(t12);
    btod_random<2>().perform(t21);
    btod_random<2>().perform(t22);
    t11.set_immutable();
    t12.set_immutable();
    t21.set_immutable();
    t22.set_immutable();

    btod_add<2> add1(t11, 1.0);
    add1.add_op(t12, 1.0);
    add1.perform(t1);
    btod_add<2> add2(t21, 1.0);
    add2.add_op(t22, 1.0);
    add2.perform(t2);
    btod_mult<2>(t1, t2, false).perform(t3_ref);

    letter i, a;
    t3(i|a) = mult(t11(i|a) + t12(i|a), t21(i|a) + t22(i|a));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void mult_test::test_ee_1b() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_ee_1b()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);

    btensor<2> t11(sp_ia), t12(sp_ia), t1(sp_ia);
    btensor<2> t21(sp_ia), t22(sp_ia), t2(sp_ia);
    btensor<2> t3(sp_ia), t3_ref(sp_ia);

    btod_random<2>().perform(t11);
    btod_random<2>().perform(t12);
    btod_random<2>().perform(t21);
    btod_random<2>().perform(t22);
    t11.set_immutable();
    t12.set_immutable();
    t21.set_immutable();
    t22.set_immutable();

    btod_add<2> add1(t11, 1.0);
    add1.add_op(t12, 1.0);
    add1.perform(t1);
    btod_add<2> add2(t21, 1.0);
    add2.add_op(t22, 1.0);
    add2.perform(t2);
    btod_mult<2>(t1, t2, true).perform(t3_ref);

    letter i, a;
    t3(i|a) = div(t11(i|a) + t12(i|a), t21(i|a) + t22(i|a));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void mult_test::test_ee_2() throw(libtest::test_exception) {

    static const char *testname = "mult_test::test_ee_2()";

    try {

    bispace<1> sp_i(10);
    bispace<3> sp_ijk(sp_i&sp_i&sp_i);

    btensor<3> t11(sp_ijk), t12(sp_ijk), t1(sp_ijk);
    btensor<3> t21(sp_ijk), t22(sp_ijk), t2(sp_ijk);
    btensor<3> t3(sp_ijk), t3_ref(sp_ijk);

    btod_random<3>().perform(t11);
    btod_random<3>().perform(t12);
    btod_random<3>().perform(t21);
    btod_random<3>().perform(t22);
    t11.set_immutable();
    t12.set_immutable();
    t21.set_immutable();
    t22.set_immutable();

    btod_add<3> add1(t11, 1.0);
    add1.add_op(t12, 1.0);
    add1.perform(t1);
    btod_add<3> add2(t21, 1.0);
    add2.add_op(t22, 1.0);
    add2.perform(t2);

    permutation<3> perm1, perm2;
    perm1.permute(1, 2).permute(0, 1);
    perm2.permute(0, 1).permute(1, 2);
    btod_mult<3>(t1, perm1, t2, perm2, false).perform(t3_ref);

    letter i, j, k;
    t3(i|j|k) = mult(t11(j|k|i) + t12(j|k|i), t21(k|i|j) + t22(k|i|j));

    compare_ref<3>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
