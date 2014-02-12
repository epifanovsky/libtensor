#include <libtensor/exception.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/iface/btensor.h>
#include "node_ident_any_tensor_test.h"

namespace libtensor {


void node_ident_any_tensor_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
}


namespace {

class tensor_i {
public:
    bool equals(const tensor_i &other) const {
        return this == &other;
    }
};

class tensor : public tensor_i {

};

} // unnamed namespace


void node_ident_any_tensor_test::test_1() {

    static const char testname[] = "node_ident_any_tensor_test::test_1()";

    try {

    tensor t;
    tensor_i &ti = t;
    any_tensor<1, int> tt1(ti);
    any_tensor<2, double> tt2(ti);

    expr::node_ident_any_tensor<1, int> i1(tt1);
    expr::node_ident_any_tensor<2, double> i2(tt2);

    expr::node_ident &i1b = i1, &i2b = i2;

    if(i1b.get_n() != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order (1).");
    }
    if(i1b.get_type() != typeid(int)) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor type (1).");
    }
    if(i2b.get_n() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order (2).");
    }
    if(i2b.get_type() != typeid(double)) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor type (2).");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void node_ident_any_tensor_test::test_2() {

    static const char testname[] = "node_ident_any_tensor_test::test_2()";

    try {

    tensor t1, t2;
    tensor_i &t1i = t1, &t2i = t2;

    any_tensor<1, double> at1(t1);
    any_tensor<2, double> at2(t2);

    expr::node_ident_any_tensor<1, double> i1(at1);
    expr::node_ident_any_tensor<2, double> i2(at2);
    expr::node_ident_any_tensor<2, double> i3(at2);

    expr::node_ident &i1b = i1, &i2b = i2, &i3b = i3;

    if(i1b.get_n() != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order (1).");
    }
    if(i1b.get_type() != typeid(double)) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor type (1).");
    }
    if(i2b.get_n() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order (2).");
    }
    if(i2b.get_type() != typeid(double)) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor type (2).");
    }
    if(i3b.get_n() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order (3).");
    }
    if(i3b.get_type() != typeid(double)) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor type (3).");
    }

    if(i1b == i2b) {
        fail_test(testname, __FILE__, __LINE__, "i1 == i2.");
    }
    if(i1b == i3b) {
        fail_test(testname, __FILE__, __LINE__, "i1 == i3.");
    }
    if(! (i2b == i3b)) {
        fail_test(testname, __FILE__, __LINE__, "i2 != i3.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
