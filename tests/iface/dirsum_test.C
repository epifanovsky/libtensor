#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/btod_add.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_dirsum.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/block_tensor/btod_set.h>
#include <libtensor/libtensor.h>
#include "../compare_ref.h"
#include "dirsum_test.h"

namespace libtensor {


void dirsum_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);

    try {
        test_tt_1();
        test_tt_2();
        test_tt_3();
        test_tt_4();
        test_tt_5();
        test_tt_6();
        test_te_1();
        test_et_1();
        test_ee_1();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void dirsum_test::test_tt_1() throw(libtest::test_exception) {

    static const char *testname = "dirsum_test::test_tt_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);

    btensor<1> t1(sp_i);
    btensor<1> t2(sp_a);
    btensor<2> t3(sp_ia), t3_ref(sp_ia);

    btod_random<1>().perform(t1);
    btod_random<1>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    btod_dirsum<1, 1>(t1, 1.0, t2, 1.0).perform(t3_ref);

    letter i, a;
    t3(i|a) = dirsum(t1(i), t2(a));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void dirsum_test::test_tt_2() throw(libtest::test_exception) {

    static const char *testname = "dirsum_test::test_tt_2()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ij(sp_i&sp_i), sp_ab(sp_a&sp_a);
    bispace<4> sp_ijab((sp_i&sp_i)|(sp_a&sp_a));

    btensor<2> t1(sp_ij);
    btensor<2> t2(sp_ab);
    btensor<4> t3(sp_ijab), t3_ref(sp_ijab);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    btod_dirsum<2, 2>(t1, 1.0, t2, 1.0).perform(t3_ref);

    letter i, j, a, b;
    t3(i|j|a|b) = dirsum(t1(i|j), t2(a|b));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void dirsum_test::test_tt_3() throw(libtest::test_exception) {

    static const char *testname = "dirsum_test::test_tt_3()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<1> sp_j(sp_i), sp_b(sp_a);
    bispace<2> sp_ij(sp_i&sp_j), sp_ab(sp_a&sp_b);
    bispace<4> sp_iajb(sp_i|sp_a|sp_j|sp_b, (sp_i&sp_j)|(sp_a&sp_b));

    btensor<2> t1(sp_ij);
    btensor<2> t2(sp_ab);
    btensor<4> t3(sp_iajb), t3_ref(sp_iajb);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    permutation<4> perm3;
    perm3.permute(1, 2); // ijab->iajb
    btod_dirsum<2, 2>(t1, 1.0, t2, 1.0, perm3).perform(t3_ref);

    letter i, j, a, b;
    t3(i|a|j|b) = dirsum(t1(i|j), t2(a|b));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void dirsum_test::test_tt_4() throw(libtest::test_exception) {

    static const char *testname = "dirsum_test::test_tt_4()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ij(sp_i&sp_i), sp_ab(sp_a&sp_a);
    bispace<4> sp_ijab((sp_i&sp_i)|(sp_a&sp_a));

    btensor<2> t1(sp_ij);
    btensor<2> t2(sp_ab);
    btensor<4> t3(sp_ijab), t3_ref(sp_ijab);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    btod_dirsum<2, 2>(t1, 2.0, t2, -1.0).perform(t3_ref);

    letter i, j, a, b;
    t3(i|j|a|b) = dirsum(2.0 * t1(i|j), -t2(a|b));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void dirsum_test::test_tt_5() throw(libtest::test_exception) {

    static const char *testname = "dirsum_test::test_tt_5()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<1> sp_j(sp_i), sp_b(sp_a);
    bispace<2> sp_ij(sp_i&sp_j), sp_ab(sp_a&sp_b);
    bispace<4> sp_iajb(sp_i|sp_a|sp_j|sp_b, (sp_i&sp_j)|(sp_a&sp_b));

    btensor<2> t1(sp_ij);
    btensor<2> t2(sp_ab);
    btensor<4> t3(sp_iajb), t3_ref(sp_iajb);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    permutation<4> perm3;
    perm3.permute(1, 2); // ijab->iajb
    btod_dirsum<2, 2>(t1, -1.5, t2, 1.0, perm3).perform(t3_ref);

    letter i, j, a, b;
    t3(i|a|j|b) = 0.5 * dirsum(-3.0 * t1(i|j), 2.0 * t2(a|b));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void dirsum_test::test_tt_6() throw(libtest::test_exception) {

    static const char *testname = "dirsum_test::test_tt_6()";

    try {

    bispace<1> sp_i(10);
    sp_i.split(5);
    bispace<2> sp_ij(sp_i&sp_i);

    btensor<1> t1(sp_i);
    btensor<2> t2(sp_ij), t2_ref(sp_ij);

    btod_random<1>().perform(t1);
    t1.set_immutable();

    btod_dirsum<1, 1>(t1, 2.0, t1, -2.0).perform(t2_ref);

    letter i, j;
    t2(i|j) = dirsum(2.0 * t1(i), -2.0 * t1(j));

    compare_ref<2>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}



void dirsum_test::test_te_1() throw(libtest::test_exception) {

    static const char *testname = "dirsum_test::test_te_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);

    btensor<1> t1(sp_i);
    btensor<1> t21(sp_a), t22(sp_a), t2(sp_a);
    btensor<2> t3(sp_ia), t3_ref(sp_ia);

    btod_random<1>().perform(t1);
    btod_random<1>().perform(t21);
    btod_random<1>().perform(t22);
    t1.set_immutable();
    t21.set_immutable();
    t22.set_immutable();

    btod_add<1> add2(t21, 1.0);
    add2.add_op(t22, 1.0);
    add2.perform(t2);
    t2.set_immutable();

    btod_set<2>(1.0).perform(t3);
    btod_set<2>(1.0).perform(t3_ref);
    //~ btod_copy<2>(t3_ref, -1.0).perform(t3, 1.0);
    //~ btod_copy<2>(t3).perform(t3_ref);

    btod_dirsum<1, 1>(t1, 1.0, t2, 1.0).perform(t3_ref);

    letter i, a;
    t3(i|a) = dirsum(t1(i), t21(a) + t22(a));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void dirsum_test::test_et_1() throw(libtest::test_exception) {

    static const char *testname = "dirsum_test::test_et_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);

    btensor<1> t11(sp_i), t12(sp_i), t1(sp_i);
    btensor<1> t2(sp_a);
    btensor<2> t3(sp_ia), t3_ref(sp_ia);

    btod_random<1>().perform(t11);
    btod_random<1>().perform(t12);
    btod_random<1>().perform(t2);
    t11.set_immutable();
    t12.set_immutable();
    t2.set_immutable();

    btod_add<1> add1(t11, 1.0);
    add1.add_op(t12, 1.0);
    add1.perform(t1);
    btod_dirsum<1, 1>(t1, 1.0, t2, 1.0).perform(t3_ref);

    letter i, a;
    t3(i|a) = dirsum(t11(i) + t12(i), t2(a));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void dirsum_test::test_ee_1() throw(libtest::test_exception) {

    static const char *testname = "dirsum_test::test_ee_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);

    btensor<1> t11(sp_i), t12(sp_i), t1(sp_i);
    btensor<1> t21(sp_a), t22(sp_a), t2(sp_a);
    btensor<2> t3(sp_ia), t3_ref(sp_ia);

    btod_random<1>().perform(t11);
    btod_random<1>().perform(t12);
    btod_random<1>().perform(t21);
    btod_random<1>().perform(t22);
    t11.set_immutable();
    t12.set_immutable();
    t21.set_immutable();
    t22.set_immutable();

    btod_add<1> add1(t11, 1.0);
    add1.add_op(t12, 1.0);
    add1.perform(t1);
    btod_add<1> add2(t21, 1.0);
    add2.add_op(t22, 1.0);
    add2.perform(t2);
    btod_dirsum<1, 1>(t1, 1.0, t2, 1.0).perform(t3_ref);

    letter i, a;
    t3(i|a) = dirsum(t11(i) + t12(i), t21(a) + t22(a));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
