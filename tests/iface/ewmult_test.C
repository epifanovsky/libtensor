#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_ewmult2.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/iface/iface.h>
#include "../compare_ref.h"
#include "ewmult_test.h"

namespace libtensor {


void ewmult_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);

    try {

        test_tt_1();
        test_tt_2();
        test_tt_3();
        test_te_1();
        test_et_1();
        test_ee_1();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void ewmult_test::test_tt_1() throw(libtest::test_exception) {

    const char testname[] = "ewmult_test::test_tt_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<3> sp_bci(sp_a|sp_a|sp_i);
    bispace<4> sp_abci(sp_a|sp_a|sp_a|sp_i);

    btensor<2> t1(sp_ia);
    btensor<3> t2(sp_bci);
    btensor<4> t3(sp_abci), t3_ref(sp_abci);

    btod_random<2>().perform(t1);
    btod_random<3>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    permutation<2> perm1;
    perm1.permute(0, 1);
    permutation<3> perm2;
    permutation<4> perm3;
    btod_ewmult2<1, 2, 1>(t1, perm1, t2, perm2, perm3).perform(t3_ref);

    letter a, b, c, i;
    t3(a|b|c|i) = ewmult(i, t1(i|a), t2(b|c|i));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ewmult_test::test_tt_2() throw(libtest::test_exception) {

    const char testname[] = "ewmult_test::test_tt_2()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<3> sp_iab(sp_i|sp_a|sp_a);

    btensor<2> t1(sp_ia);
    btensor<2> t2(sp_ia);
    btensor<3> t3(sp_iab), t3_ref(sp_iab);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    permutation<2> perm1;
    perm1.permute(0, 1);
    permutation<2> perm2;
    perm2.permute(0, 1);
    permutation<3> perm3;
    perm3.permute(1, 2).permute(0, 1);
    btod_ewmult2<1, 1, 1>(t1, perm1, t2, perm2, perm3).perform(t3_ref);

    letter a, b, i;
    t3(i|a|b) = ewmult(i, t1(i|a), t2(i|b));

    compare_ref<3>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ewmult_test::test_tt_3() throw(libtest::test_exception) {

    const char testname[] = "ewmult_test::test_tt_3()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<3> sp_bci(sp_a|sp_a|sp_i);
    bispace<4> sp_abci(sp_a|sp_a|sp_a|sp_i);

    btensor<2> t1(sp_ia);
    btensor<3> t2(sp_bci);
    btensor<4> t3(sp_abci), t3_ref(sp_abci);

    btod_random<2>().perform(t1);
    btod_random<3>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    permutation<2> perm1;
    perm1.permute(0, 1);
    permutation<3> perm2;
    permutation<4> perm3;
    btod_ewmult2<1, 2, 1>(t1, perm1, t2, perm2, perm3).perform(t3_ref);

    letter a, b, c, i;
    t3(a|b|c|i) = -ewmult(i, 2.0 * t1(i|a), -0.5 * t2(b|c|i));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ewmult_test::test_te_1() throw(libtest::test_exception) {

    const char testname[] = "ewmult_test::test_te_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<3> sp_bci(sp_a|sp_a|sp_i), sp_icb(sp_i|sp_a|sp_a);
    bispace<4> sp_abci(sp_a|sp_a|sp_a|sp_i);

    btensor<2> t1(sp_ia);
    btensor<3> t2a(sp_bci), t2b(sp_icb), t2(sp_bci);
    btensor<4> t3(sp_abci), t3_ref(sp_abci);

    btod_random<2>().perform(t1);
    btod_random<3>().perform(t2a);
    btod_random<3>().perform(t2b);
    t1.set_immutable();
    t2a.set_immutable();
    t2b.set_immutable();

    btod_copy<3>(t2a).perform(t2);
    btod_copy<3>(t2b, permutation<3>().permute(0, 2), -1.0).perform(t2, 1.0);
    permutation<2> perm1;
    perm1.permute(0, 1);
    permutation<3> perm2;
    permutation<4> perm3;
    btod_ewmult2<1, 2, 1>(t1, perm1, t2, perm2, perm3).perform(t3_ref);

    letter a, b, c, i;
    t3(a|b|c|i) = ewmult(i, t1(i|a), t2a(b|c|i) - t2b(i|c|b));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ewmult_test::test_et_1() throw(libtest::test_exception) {

    const char testname[] = "ewmult_test::test_et_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a), sp_ai(sp_a|sp_i);
    bispace<3> sp_bci(sp_a|sp_a|sp_i);
    bispace<4> sp_abci(sp_a|sp_a|sp_a|sp_i);

    btensor<2> t1a(sp_ai), t1b(sp_ia), t1(sp_ia);
    btensor<3> t2(sp_bci);
    btensor<4> t3(sp_abci), t3_ref(sp_abci);

    btod_random<2>().perform(t1a);
    btod_random<2>().perform(t1b);
    btod_random<3>().perform(t2);
    t1a.set_immutable();
    t1b.set_immutable();
    t2.set_immutable();

    btod_copy<2>(t1a, permutation<2>().permute(0, 1), 0.5).perform(t1);
    btod_copy<2>(t1b, permutation<2>(), 1.5).perform(t1, 1.0);
    permutation<2> perm1;
    perm1.permute(0, 1);
    permutation<3> perm2;
    permutation<4> perm3;
    btod_ewmult2<1, 2, 1>(t1, perm1, t2, perm2, perm3).perform(t3_ref);

    letter a, b, c, i;
    t3(a|b|c|i) = ewmult(i, 0.5 * t1a(a|i) + 1.5 * t1b(i|a), t2(b|c|i));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ewmult_test::test_ee_1() throw(libtest::test_exception) {

    const char testname[] = "ewmult_test::test_ee_1()";

    try {

    bispace<1> sp_i(10), sp_a(20);
    bispace<2> sp_ia(sp_i|sp_a), sp_ai(sp_a|sp_i);
    bispace<3> sp_bci(sp_a|sp_a|sp_i), sp_ibc(sp_i|sp_a|sp_a);
    bispace<4> sp_abci(sp_a|sp_a|sp_a|sp_i);

    btensor<2> t1a(sp_ia), t1b(sp_ai), t1(sp_ai);
    btensor<3> t2a(sp_bci), t2b(sp_ibc), t2(sp_bci);
    btensor<4> t3(sp_abci), t3_ref(sp_abci);

    btod_random<2>().perform(t1a);
    btod_random<2>().perform(t1b);
    btod_random<3>().perform(t2a);
    btod_random<3>().perform(t2b);
    t1a.set_immutable();
    t1b.set_immutable();
    t2a.set_immutable();
    t2b.set_immutable();

    btod_copy<2>(t1a, permutation<2>().permute(0, 1), 1.0).perform(t1);
    btod_copy<2>(t1b, permutation<2>(), 1.0).perform(t1, 1.0);
    btod_copy<3>(t2a, permutation<3>(), 1.0).perform(t2);
    btod_copy<3>(t2b, permutation<3>().permute(0, 1).permute(1, 2), 1.0).
        perform(t2, 1.0);
    permutation<2> perm1;
    permutation<3> perm2;
    permutation<4> perm3;
    btod_ewmult2<1, 2, 1>(t1, perm1, t2, perm2, perm3).perform(t3_ref);

    letter a, b, c, i;
    t3(a|b|c|i) = ewmult(i, t1a(i|a) + t1b(a|i), t2a(b|c|i) + t2b(i|b|c));

    compare_ref<4>::compare(testname, t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
