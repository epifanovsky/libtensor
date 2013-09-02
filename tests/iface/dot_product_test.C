#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/iface/iface.h>
#include "dot_product_test.h"

namespace libtensor {


void dot_product_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);

    try {

        test_tt_ij_ij_1();
        test_tt_ij_ji_1();
        test_te_ij_ij_1();
        test_te_ij_ji_1();
        test_et_1();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void dot_product_test::test_tt_ij_ij_1() throw(libtest::test_exception) {

    static const char *testname = "dot_product_test::test_tt_ij_ij_1()";

    try {

    bispace<1> si(10), sj(11);
    bispace<2> sij(si|sj);
    btensor<2> bt1(sij), bt2(sij);

    btod_random<2>().perform(bt1);
    btod_random<2>().perform(bt2);
    double c_ref = btod_dotprod<2>(bt1, bt2).calculate();

    letter i, j;
    double c = dot_product(bt1(i|j), bt2(i|j));
    check_ref(testname, c, c_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


void dot_product_test::test_tt_ij_ji_1() throw(libtest::test_exception) {

    static const char *testname = "dot_product_test::test_tt_ij_ji_1()";

    try {

    bispace<1> si(10), sj(11);
    bispace<2> sij(si|sj), sji(sj|si);
    btensor<2> bt1(sij), bt2(sji);

    btod_random<2>().perform(bt1);
    btod_random<2>().perform(bt2);
    permutation<2> p1, p2;
    p2.permute(0, 1);
    double c_ref = btod_dotprod<2>(bt1, p1, bt2, p2).calculate();

    letter i, j;
    double c = dot_product(bt1(i|j), bt2(j|i));
    check_ref(testname, c, c_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void dot_product_test::test_te_ij_ij_1() throw(libtest::test_exception) {

    static const char *testname = "dot_product_test::test_te_ij_ij_1()";

    try {

    bispace<1> si(10), sj(11);
    bispace<2> sij(si|sj);
    btensor<2> bt1(sij), bt2(sij), bt3(sij), bt4(sij);

    btod_random<2>().perform(bt1);
    btod_random<2>().perform(bt2);
    btod_random<2>().perform(bt3);
    btod_copy<2>(bt2).perform(bt4);
    btod_copy<2>(bt3).perform(bt4, 0.5);
    double c_ref = btod_dotprod<2>(bt1, bt4).calculate();

    letter i, j;
    double c = dot_product(bt1(i|j), bt2(i|j) + 0.5 * bt3(i|j));
    check_ref(testname, c, c_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }

}


void dot_product_test::test_te_ij_ji_1() throw(libtest::test_exception) {

    static const char *testname = "dot_product_test::test_te_ij_ji_1()";

    try {

    bispace<1> si(10), sj(11);
    bispace<2> sij(si|sj), sji(sj|si);
    btensor<2> bt1(sij), bt2(sij), bt3(sji), bt4(sij);

    btod_random<2>().perform(bt1);
    btod_random<2>().perform(bt2);
    btod_random<2>().perform(bt3);
    permutation<2> perm;
    perm.permute(0, 1);
    btod_copy<2>(bt2).perform(bt4);
    btod_copy<2>(bt3, perm).perform(bt4, 0.5);
    double c_ref = btod_dotprod<2>(bt1, bt4).calculate();

    letter i, j;
    double c = dot_product(bt1(i|j), bt2(i|j) + 0.5 * bt3(j|i));
    check_ref(testname, c, c_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void dot_product_test::test_et_1() throw(libtest::test_exception) {

    static const char *testname = "dot_product_test::test_et_1()";

    try {

    bispace<1> si(10), sa(20);
    si.split(5);
    sa.split(10);
    bispace<2> sia(si|sa);
    bispace<4> sijab((si&si)|(sa&sa));
    btensor<2> bt1(sia), bt1a(sia);
    btensor<4> bt2(sijab);

    btod_random<2>().perform(bt1);
    btod_random<4>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    contraction2<2, 0, 2> contr;
    contr.contract(1, 0); contr.contract(3, 1);
    btod_contract2<2, 0, 2>(contr, bt2, bt1).perform(bt1a);
    double c_ref = btod_dotprod<2>(bt1, bt1a).calculate();

    letter i, j, a, b;
    double c = dot_product(contract(j|b, bt2(i|j|a|b), bt1(j|b)), bt1(i|a));
    check_ref(testname, c, c_ref);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void dot_product_test::check_ref(const char *testname, double d, double d_ref)
    throw(libtest::test_exception) {

    if(fabs(d - d_ref) > fabs(d_ref * 1e-14)) {
        std::ostringstream ss;
        ss << "Result doesn't match reference: " << d << " (res), "
            << d_ref << " (ref), " << d - d_ref << " (diff)";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
}



} // namespace libtensor
