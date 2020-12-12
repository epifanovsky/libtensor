#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/btod_add.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_scale.h>
#include <libtensor/block_tensor/btod_set.h>
#include <libtensor/block_tensor/btod_set_diag.h>
#include <libtensor/block_tensor/btod_shift_diag.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/libtensor.h>
#include "../compare_ref.h"
#include "set_test.h"

namespace libtensor {


void set_test::perform() throw(libtest::test_exception) {

    allocator<double>::init();

    try {

        test_s_1(0.0);
        test_s_1(1.0);
        test_s_2(0.0);
        test_s_2(-2.1);
        test_d_1(0.0);
        test_d_1(1.0);
        test_d_2(0.0);
        test_d_2(-1.0);
        test_x_1(0.0);
        test_x_1(0.2);
        test_x_2(0.0);
        test_x_2(0.1);

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void set_test::test_s_1(double d) {

    std::ostringstream tnss;
    tnss << "set_test::test_s_1(" << d << ")";
    std::string tn = tnss.str();
    const char *testname = tn.c_str();

    try {

    bispace<1> sp_i(10), sp_a(11);
    bispace<2> sp_ij(sp_i&sp_i);

    btensor<2> t1(sp_ij), t2(sp_ij), t2_ref(sp_ij);

    btod_random<2>().perform(t1);
    t1.set_immutable();

    btod_set<2>(d).perform(t2_ref);

    letter i, j;
    t2(i|j) = set(d, t1(i|j));

    compare_ref<2>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void set_test::test_s_2(double d) {

    std::ostringstream tnss;
    tnss << "set_test::test_s_2(" << d << ")";
    std::string tn = tnss.str();
    const char *testname = tn.c_str();

    try {

    bispace<1> sp_i(10), sp_a(11);
    bispace<3> sp_ija((sp_i&sp_i)|sp_a), sp_iaj(sp_i|sp_a|sp_i);

    btensor<3> t1(sp_ija), t2(sp_iaj), t2_ref(sp_iaj);

    btod_random<3>().perform(t1);
    t1.set_immutable();

    btod_set<3>(d).perform(t2_ref);

    letter i, j, a;
    t2(i|a|j) = set(d, t1(i|j|a));

    compare_ref<3>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void set_test::test_d_1(double d) {

    std::ostringstream tnss;
    tnss << "set_test::test_d_1(" << d << ")";
    std::string tn = tnss.str();
    const char *testname = tn.c_str();

    try {

    bispace<1> sp_i(10);
    bispace<2> sp_ij(sp_i&sp_i);

    btensor<2> t1(sp_ij), t2(sp_ij), t2_ref(sp_ij);

    btod_random<2>().perform(t1);
    t1.set_immutable();

    btod_copy<2>(t1, permutation<2>().permute(0, 1)).perform(t2_ref);

    sequence<2, size_t> msk(0);
    msk[0] = 1; msk[1] = 1;
    btod_set_diag<2>(msk, d).perform(t2_ref);

    letter i, j;
    t2(j|i) = set(i, i|j, d, t1(i|j));

    compare_ref<2>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void set_test::test_d_2(double d) {

    std::ostringstream tnss;
    tnss << "set_test::test_d_2(" << d << ")";
    std::string tn = tnss.str();
    const char *testname = tn.c_str();

    try {

    bispace<1> sp_i(10), sp_a(5), sp_p(7);
    bispace<5> sp_iajbp(((sp_i|sp_a)&(sp_i|sp_a))|sp_p);
    bispace<5> sp_ijpab((sp_i&sp_i)|sp_p|(sp_a&sp_a));

    btensor<5> t1(sp_iajbp), t2(sp_ijpab), t2_ref(sp_ijpab);

    btod_random<5>().perform(t1);
    t1.set_immutable();

    permutation<5> perm;
    perm.permute(1, 2).permute(3, 4).permute(2, 3);
    btod_copy<5>(t1, perm).perform(t2_ref);

    sequence<5, size_t> msk(0);
    msk[0] = 1; msk[1] = 1; msk[3] = 2; msk[4] = 2;
    btod_set_diag<5>(msk, d).perform(t2_ref);

    letter i, j, a, b, p;
    t2(i|j|p|a|b) = set(i|a, i|j|a|b, d, t1(i|a|j|b|p));

    compare_ref<5>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void set_test::test_x_1(double d) {

    std::ostringstream tnss;
    tnss << "set_test::test_x_1(" << d << ")";
    std::string tn = tnss.str();
    const char *testname = tn.c_str();

    try {

    bispace<1> sp_i(10), sp_a(4);
    bispace<3> sp_iaj(sp_i|sp_a|sp_i);
    btensor<3> t1(sp_iaj), t2(sp_iaj), t2_ref(sp_iaj);

    btod_random<3>().perform(t1);
    t1.set_immutable();

    btod_copy<3>(t1).perform(t2_ref);
    sequence<3, size_t> msk(0);
    msk[0] = 1; msk[2] = 1;
    btod_shift_diag<3>(msk, d).perform(t2_ref);
    btod_scale<3>(t2_ref, 0.5).perform();

    letter i, j, a;
    t2(i|a|j) = 0.5 * shift(i, i|j, d, t1(i|a|j));

    compare_ref<3>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void set_test::test_x_2(double d) {

    std::ostringstream tnss;
    tnss << "set_test::test_x_2(" << d << ")";
    std::string tn = tnss.str();
    const char *testname = tn.c_str();

    try {

    bispace<1> sp_i(10), sp_a(16), sp_p(4);
    sp_i.split(5);
    sp_a.split(8);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<6> sp_piajbq(sp_p|(sp_ia&sp_ia)|sp_p);
    bispace<6> sp_ijabpq((sp_i&sp_i)|(sp_a&sp_a)|(sp_p&sp_p));

    btensor<6> t1(sp_piajbq), t2(sp_ijabpq), t2_ref(sp_ijabpq);

    btod_random<6>().perform(t1);
    t1.set_immutable();

    permutation<6> perm;
    perm.permute(0, 4).permute(0, 3).permute(0, 1);
    btod_copy<6>(t1, perm).perform(t2_ref);

    sequence<6, size_t> msk(0);
    msk[0] = 1; msk[1] = 1; msk[2] = 2; msk[3] = 2;
    btod_shift_diag<6>(msk, d).perform(t2_ref);

    letter i, j, a, b, p, q;
    t2(i|j|a|b|p|q) = shift(i|a, i|j|a|b, d, t1(p|i|a|j|b|q));

    compare_ref<6>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
