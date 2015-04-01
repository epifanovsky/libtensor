#include <libtensor/libtensor.h>
#include <libtensor/core/allocator.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include <libtensor/expr/ctf_btensor/ctf_btensor.h>
#include "../compare_ref.h"
#include "ctf_trace_test.h"

namespace libtensor {


void ctf_trace_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);
    ctf::init();

    try {

        test_t_1();
        test_t_2();
        test_t_3();
        test_e_1();
        test_e_2();
        test_e_3();

    } catch(...) {
        allocator<double>::shutdown();
        ctf::exit();
        throw;
    }

    ctf::exit();
    allocator<double>::shutdown();
}


void ctf_trace_test::test_t_1() {

    static const char testname[] = "ctf_trace_test::test_t_1()";

    try {

    bispace<1> si(10);
    si.split(5).split(7);
    bispace<2> sij(si&si);

    btensor<2, double> t(sij);
    ctf_btensor<2, double> dt(sij);

    btod_random<2>().perform(t);
    t.set_immutable();

    ctf_btod_distribute<2>(t).perform(dt);

    letter i, j;

    double d_ref = trace(i, j, t(i|j));
    double d = trace(i, j, dt(i|j));

    compare_ref<1>::compare(testname, d, d_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


void ctf_trace_test::test_t_2() {

    static const char testname[] = "ctf_trace_test::test_t_2()";

    try {

    bispace<1> si(10), sj(11);
    si.split(5).split(7);
    sj.split(3).split(6);
    bispace<4> sijkl((si|sj)&(si|sj));

    btensor<4, double> t(sijkl);
    ctf_btensor<4, double> dt(sijkl);

    btod_random<4>().perform(t);
    t.set_immutable();

    ctf_btod_distribute<4>(t).perform(dt);

    letter i, j, k, l;

    double d_ref = trace(i|j, k|l, t(i|j|k|l));
    double d = trace(i|j, k|l, dt(i|j|k|l));

    compare_ref<1>::compare(testname, d, d_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


void ctf_trace_test::test_t_3() {

    static const char testname[] = "ctf_trace_test::test_t_3()";

    try {

    bispace<1> si(10);
    si.split(5).split(7);
    bispace<4> sijkl(si&si&si&si);

    btensor<4, double> t(sijkl);
    ctf_btensor<4, double> dt(sijkl);

    btod_random<4>().perform(t);
    t.set_immutable();

    ctf_btod_distribute<4>(t).perform(dt);

    letter i, j, k, l;

    double d_ref = trace(i|k, j|l, t(i|j|k|l));
    double d = trace(i|k, j|l, dt(i|j|k|l));

    compare_ref<1>::compare(testname, d, d_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


void ctf_trace_test::test_e_1() {

    static const char testname[] = "ctf_trace_test::test_e_1()";

    try {

    bispace<1> si(10);
    si.split(5).split(7);
    bispace<2> sij(si&si);

    btensor<2, double> t1(sij), t2(sij);
    ctf_btensor<2, double> dt1(sij), dt2(sij);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    ctf_btod_distribute<2>(t1).perform(dt1);
    ctf_btod_distribute<2>(t2).perform(dt2);

    letter i, j;

    double d_ref = trace(i, j, t1(i|j) + t2(i|j));
    double d = trace(i, j, dt1(i|j) + dt2(i|j));

    compare_ref<1>::compare(testname, d, d_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


void ctf_trace_test::test_e_2() {

    static const char testname[] = "ctf_trace_test::test_e_2()";

    try {

    bispace<1> si(10);
    si.split(5).split(7);
    bispace<2> sij(si&si);

    btensor<2, double> t1(sij), t2(sij);
    ctf_btensor<2, double> dt1(sij), dt2(sij);

    btod_random<2>().perform(t1);
    btod_random<2>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    ctf_btod_distribute<2>(t1).perform(dt1);
    ctf_btod_distribute<2>(t2).perform(dt2);

    letter i, j;

    double d_ref = trace(i, j, t1(j|i) + t2(i|j));
    double d = trace(i, j, dt1(j|i) + dt2(i|j));

    compare_ref<1>::compare(testname, d, d_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


void ctf_trace_test::test_e_3() {

    static const char testname[] = "ctf_trace_test::test_e_3()";

    try {

    bispace<1> si(10), sj(11);
    si.split(5).split(7);
    sj.split(3).split(6);
    bispace<4> sijkl((si|sj)&(si|sj));

    btensor<4, double> t1(sijkl), t2(sijkl);
    ctf_btensor<4, double> dt1(sijkl), dt2(sijkl);

    btod_random<4>().perform(t1);
    btod_random<4>().perform(t2);
    t1.set_immutable();
    t2.set_immutable();

    ctf_btod_distribute<4>(t1).perform(dt1);
    ctf_btod_distribute<4>(t2).perform(dt2);

    letter i, j, k, l;

    double d_ref = trace(i|j, k|l, t1(i|j|k|l) + t2(i|j|k|l));
    double d = trace(i|j, k|l, dt1(i|j|k|l) + dt2(i|j|k|l));

    compare_ref<1>::compare(testname, d, d_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


} // namespace libtensor
