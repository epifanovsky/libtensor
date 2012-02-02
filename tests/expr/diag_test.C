#include <libtensor/btod/btod_add.h>
#include <libtensor/btod/btod_copy.h>
#include <libtensor/btod/btod_diag.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/btensor/btensor.h>
#include <libtensor/expr/assignment_operator.h>
#include <libtensor/expr/operators.h>
#include "../compare_ref.h"
#include "diag_test.h"

namespace libtensor {


void diag_test::perform() throw(libtest::test_exception) {

    allocator<double>::vmm().init(16, 16, 16777216, 16777216);

    try {

        test_t_1();
        test_t_2();
        test_t_3();
        test_t_4();
        test_e_1();
        test_x_1();

    } catch(...) {
        allocator<double>::vmm().shutdown();
        throw;
    }

    allocator<double>::vmm().shutdown();
}


void diag_test::test_t_1() throw(libtest::test_exception) {

    static const char *testname = "diag_test::test_t_1()";

    try {

    bispace<1> sp_i(10);
    bispace<2> sp_ij(sp_i&sp_i);

    btensor<2> t1(sp_ij);
    btensor<1> t2(sp_i), t2_ref(sp_i);

    btod_random<2>().perform(t1);
    t1.set_immutable();

    mask<2> msk;
    msk[0] = true; msk[1] = true;
    permutation<1> perm;
    btod_diag<2, 2>(t1, msk, perm).perform(t2_ref);

    letter i, j, k;
    t2(k) = diag(k, i|j, t1(j|i));

    compare_ref<1>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_test::test_t_2() throw(libtest::test_exception) {

    static const char *testname = "diag_test::test_t_2()";

    try {

    bispace<1> sp_i(10), sp_a(11);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<3> sp_ija(sp_i&sp_i|sp_a);

    btensor<3> t1(sp_ija);
    btensor<2> t2(sp_ia), t2_ref(sp_ia);

    btod_random<3>().perform(t1);
    t1.set_immutable();

    mask<3> msk;
    msk[0] = true; msk[1] = true;
    permutation<2> perm;
    btod_diag<3, 2>(t1, msk, perm).perform(t2_ref);

    letter i, j, a;
    t2(i|a) = diag(i, i|j, t1(i|j|a));

    compare_ref<2>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_test::test_t_3() throw(libtest::test_exception) {

    static const char *testname = "diag_test::test_t_3()";

    try {

    bispace<1> sp_i(10), sp_a(11), sp_j(sp_i);
    bispace<2> sp_ai(sp_a|sp_i);
    bispace<3> sp_iaj(sp_i|sp_a|sp_j, sp_i&sp_j|sp_a);

    btensor<3> t1(sp_iaj);
    btensor<2> t2(sp_ai), t2_ref(sp_ai);

    btod_random<3>().perform(t1);
    t1.set_immutable();

    mask<3> msk;
    msk[0] = true; msk[2] = true;
    permutation<2> perm;
    perm.permute(0, 1); // ia->ai
    btod_diag<3, 2>(t1, msk, perm).perform(t2_ref);

    letter i, j, a;
    t2(a|i) = diag(i, i|j, t1(i|a|j));

    compare_ref<2>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_test::test_t_4() throw(libtest::test_exception) {

    static const char *testname = "diag_test::test_t_4()";

    try {

    bispace<1> sp_i(10);
    bispace<2> sp_ij(sp_i&sp_i);

    btensor<2> t1(sp_ij);
    btensor<1> t2(sp_i), t2_ref(sp_i);

    btod_random<2>().perform(t1);
    t1.set_immutable();

    mask<2> msk;
    msk[0] = true; msk[1] = true;
    permutation<1> perm;
    btod_diag<2, 2>(t1, msk, perm, -1.0).perform(t2_ref);

    letter i, j;
    t2(i) = - diag(i, i|j, t1(i|j));

    compare_ref<1>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void diag_test::test_e_1() throw(libtest::test_exception) {

    static const char *testname = "diag_test::test_e_1()";

    try {

    bispace<1> sp_i(10);
    bispace<2> sp_ij(sp_i&sp_i);

    btensor<2> t1a(sp_ij), t1b(sp_ij);
    btensor<1> t2a(sp_i), t2b(sp_i), t2(sp_i), t2_ref(sp_i);

    btod_random<2>().perform(t1a);
    btod_random<2>().perform(t1b);
    t1a.set_immutable();
    t1b.set_immutable();

    mask<2> msk;
    msk[0] = true; msk[1] = true;
    permutation<1> perm;
    btod_diag<2, 2>(t1a, msk, perm).perform(t2a);
    btod_diag<2, 2>(t1b, msk, perm).perform(t2b);
    btod_add<1> add(t2a);
    add.add_op(t2b);
    add.perform(t2_ref);

    letter i, j, k;
    t2(k) = diag(k, i|j, t1a(j|i) + t1b(i|j));

    compare_ref<1>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

void diag_test::test_x_1() throw(libtest::test_exception) {

    static const char *testname = "diag_test::test_x_1()";

    try {

    bispace<1> sp_i(10), sp_a(16);
    sp_i.split(5);
    sp_a.split(8);
    bispace<1> sp_j(sp_i);
    bispace<2> sp_ia(sp_i|sp_a), sp_jb(sp_i|sp_a);
    bispace<3> sp_iaj(sp_i|sp_a|sp_j, (sp_i&sp_j)|sp_a);
    bispace<4> sp_iajb(sp_ia&sp_jb);

    btensor<2> t1(sp_ia), t3(sp_ia), t3_ref(sp_ia);
    btensor<4> t2(sp_iajb);
    btensor<3> tx(sp_iaj);

    btod_random<2>().perform(t1);
    t1.set_immutable();
    btod_random<4>().perform(t2);
    t2.set_immutable();

    btod_copy<2>(t1).perform(t3_ref);
    mask<4> msk1;
    msk1[1] = true; msk1[3] = true;
    permutation<3> perm1;
    btod_diag<4, 2>(t2, msk1, perm1, 1.0).perform(tx);
    mask<3> msk2;
    msk2[0] = true; msk2[2] = true;
    permutation<2> perm2;
    //perm2.permute(0, 1);
    btod_diag<3, 2>(tx, msk2, perm2, -1.0).perform(t3_ref, 1.0);

    letter i, j, a, b;
    t3(i|a) = t1(i|a) - diag(i, i|j, diag(a, a|b, t2(i|a|j|b)));

    compare_ref<2>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
