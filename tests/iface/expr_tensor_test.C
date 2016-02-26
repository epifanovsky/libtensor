#include <libtensor/exception.h>
#include <libtensor/expr/btensor/btensor.h>
#include <libtensor/expr/iface/expr_tensor.h>
#include <libtensor/expr/operators/operators.h>
#include "expr_tensor_test.h"

namespace libtensor {


namespace {

void calculate_4(expr::expr_lhs<2, double> &t3,
        any_tensor<2, double> &t1, any_tensor<2, double> &t2) {

    letter i, j, k;
    t3(i|j) = contract(k, t1(i|k), t2(j|k));
}

} // unnamed namespace


void expr_tensor_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
    test_4();
    test_5();
}


void expr_tensor_test::test_1() {

    static const char testname[] = "expr_tensor_test::test_1()";

    try {

    bispace<1> o(10);

    btensor<1> t1(o), t2(o);
    expr_tensor<1> t3;

    letter i;
    t3(i) = t1(i) + t2(i);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void expr_tensor_test::test_2() {

    static const char testname[] = "expr_tensor_test::test_2()";

    try {

    bispace<1> o(10);
    bispace<2> oo(o&o);

    btensor<2> t1(oo), t2(oo);
    expr_tensor<2> t3;
    btensor<2> t4(oo);

    letter i, j, k;
    t3(i|j) = contract(k, t1(i|k), t2(k|j));
    t4(i|j) = t3(i|j);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void expr_tensor_test::test_3() {

    static const char testname[] = "expr_tensor_test::test_3()";

    try {

    bispace<1> o(10);
    bispace<2> oo(o&o);

    btensor<2> t1(oo), t2(oo);
    expr_tensor<2> t3;
    btensor<2> t4(oo);

    letter i, j, k;
    t3(i|j) = contract(k, t1(i|k), t2(k|j));
    t4(i|j) = contract(k, t1(i|k), t3(j|k));

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void expr_tensor_test::test_4() {

    static const char testname[] = "expr_tensor_test::test_4()";

    try {

    bispace<1> o(10);
    bispace<2> oo(o&o);

    btensor<2> t1(oo), t2(oo);
    expr_tensor<2> t3;
    btensor<2> t4(oo);

    letter i, j, k;
    t3(i|j) = contract(k, t1(i|k), t2(k|j));
    calculate_4(t4, t1, t3);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void expr_tensor_test::test_5() {

    static const char testname[] = "expr_tensor_test::test_5()";

    try {

    bispace<1> o(10);

    btensor<1> t1(o), t2(o);
    expr_tensor<1> t3;

    letter i;
    t3(i) = t1(i) + t2(i);
    t3(i) *= 2.0;

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

