#include <libtensor/libtensor.h>
#include <libtensor/core/allocator.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/ctf_dense_tensor/ctf.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include <libtensor/expr/ctf_btensor/ctf_btensor.h>
#include "../compare_ref.h"
#include "ctf_set_test.h"

namespace libtensor {


void ctf_set_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);
    ctf::init();

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
        ctf::exit();
        allocator<double>::shutdown();
        throw;
    }

    ctf::exit();
    allocator<double>::shutdown();
}


void ctf_set_test::test_s_1(double d) {

    std::ostringstream tnss;
    tnss << "ctf_set_test::test_s_1(" << d << ")";
    std::string tn = tnss.str();
    const char *testname = tn.c_str();

    try {

    bispace<1> sp_i(10), sp_a(11);
    bispace<2> sp_ij(sp_i&sp_i);

    btensor<2, double> t1(sp_ij), t2(sp_ij), t2_ref(sp_ij);
    ctf_btensor<2, double> dt1(sp_ij), dt2(sp_ij);

    btod_random<2>().perform(t1);
    t1.set_immutable();

    ctf_btod_distribute<2>(t1).perform(dt1);

    letter i, j;
    t2_ref(i|j) = set(d, t1(i|j));
    dt2(i|j) = set(d, dt1(i|j));

    ctf_btod_collect<2>(dt2).perform(t2);

    compare_ref<2>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_set_test::test_s_2(double d) {

    std::ostringstream tnss;
    tnss << "ctf_set_test::test_s_2(" << d << ")";
    std::string tn = tnss.str();
    const char *testname = tn.c_str();

    try {

    bispace<1> sp_i(10), sp_a(11);
    bispace<3> sp_ija((sp_i&sp_i)|sp_a), sp_iaj(sp_i|sp_a|sp_i);

    btensor<3, double> t1(sp_ija), t2(sp_iaj), t2_ref(sp_iaj);
    ctf_btensor<3, double> dt1(sp_ija), dt2(sp_iaj);

    btod_random<3>().perform(t1);
    t1.set_immutable();

    ctf_btod_distribute<3>(t1).perform(dt1);

    letter i, j, a;
    t2_ref(i|a|j) = set(d, t1(i|j|a));
    dt2(i|a|j) = set(d, dt1(i|j|a));

    ctf_btod_collect<3>(dt2).perform(t2);

    compare_ref<3>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_set_test::test_d_1(double d) {

    std::ostringstream tnss;
    tnss << "ctf_set_test::test_d_1(" << d << ")";
    std::string tn = tnss.str();
    const char *testname = tn.c_str();

    try {

    bispace<1> sp_i(10);
    bispace<2> sp_ij(sp_i&sp_i);

    btensor<2, double> t1(sp_ij), t2(sp_ij), t2_ref(sp_ij);
    ctf_btensor<2, double> dt1(sp_ij), dt2(sp_ij);

    btod_random<2>().perform(t1);
    t1.set_immutable();

    ctf_btod_distribute<2>(t1).perform(dt1);

    letter i, j;
    t2_ref(j|i) = set(i, i|j, d, t1(i|j));
    dt2(j|i) = set(i, i|j, d, dt1(i|j));

    ctf_btod_collect<2>(dt2).perform(t2);

    compare_ref<2>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_set_test::test_d_2(double d) {

    std::ostringstream tnss;
    tnss << "ctf_set_test::test_d_2(" << d << ")";
    std::string tn = tnss.str();
    const char *testname = tn.c_str();

    try {

    bispace<1> sp_i(10), sp_a(5), sp_p(7);
    bispace<5> sp_iajbp(((sp_i|sp_a)&(sp_i|sp_a))|sp_p);
    bispace<5> sp_ijpab((sp_i&sp_i)|sp_p|(sp_a&sp_a));

    btensor<5, double> t1(sp_iajbp), t2(sp_ijpab), t2_ref(sp_ijpab);
    ctf_btensor<5, double> dt1(sp_iajbp), dt2(sp_ijpab);

    btod_random<5>().perform(t1);
    t1.set_immutable();

    ctf_btod_distribute<5>(t1).perform(dt1);

    letter i, j, a, b, p;
    t2_ref(i|j|p|a|b) = set(i|a, i|j|a|b, d, t1(i|a|j|b|p));
    dt2(i|j|p|a|b) = set(i|a, i|j|a|b, d, dt1(i|a|j|b|p));

    ctf_btod_collect<5>(dt2).perform(t2);

    compare_ref<5>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_set_test::test_x_1(double d) {

    std::ostringstream tnss;
    tnss << "ctf_set_test::test_x_1(" << d << ")";
    std::string tn = tnss.str();
    const char *testname = tn.c_str();

    try {

    bispace<1> sp_i(10), sp_a(4);
    bispace<3> sp_iaj(sp_i|sp_a|sp_i);
    btensor<3, double> t1(sp_iaj), t2(sp_iaj), t2_ref(sp_iaj);
    ctf_btensor<3, double> dt1(sp_iaj), dt2(sp_iaj);

    btod_random<3>().perform(t1);
    t1.set_immutable();

    ctf_btod_distribute<3>(t1).perform(dt1);

    letter i, j, a;
    t2_ref(i|a|j) = 0.5 * shift(i, i|j, d, t1(i|a|j));
    dt2(i|a|j) = 0.5 * shift(i, i|j, d, dt1(i|a|j));

    ctf_btod_collect<3>(dt2).perform(t2);

    compare_ref<3>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void ctf_set_test::test_x_2(double d) {

    std::ostringstream tnss;
    tnss << "ctf_set_test::test_x_2(" << d << ")";
    std::string tn = tnss.str();
    const char *testname = tn.c_str();

    try {

    bispace<1> sp_i(10), sp_a(16), sp_p(4);
    sp_i.split(5);
    sp_a.split(8);
    bispace<2> sp_ia(sp_i|sp_a);
    bispace<6> sp_piajbq(sp_p|(sp_ia&sp_ia)|sp_p);
    bispace<6> sp_ijabpq((sp_i&sp_i)|(sp_a&sp_a)|(sp_p&sp_p));

    btensor<6, double> t1(sp_piajbq), t2(sp_ijabpq), t2_ref(sp_ijabpq);
    ctf_btensor<6, double> dt1(sp_piajbq), dt2(sp_ijabpq);

    btod_random<6>().perform(t1);
    t1.set_immutable();

    ctf_btod_distribute<6>(t1).perform(dt1);

    letter i, j, a, b, p, q;
    t2_ref(i|j|a|b|p|q) = shift(i|a, i|j|a|b, d, t1(p|i|a|j|b|q));
    dt2(i|j|a|b|p|q) = shift(i|a, i|j|a|b, d, dt1(p|i|a|j|b|q));

    ctf_btod_collect<6>(dt2).perform(t2);

    compare_ref<6>::compare(testname, t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
