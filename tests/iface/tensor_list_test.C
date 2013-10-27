#include <libtensor/exception.h>
#include <libtensor/iface/btensor.h>
#include <libtensor/iface/tensor_list.h>
#include <libtensor/iface/btensor.h>
#include "tensor_list_test.h"

namespace libtensor {


void tensor_list_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
    test_3();
    test_4();
    test_5();
    test_6();
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


void tensor_list_test::test_1() {

    static const char testname[] = "tensor_list_test::test_1()";

    try {

    tensor t;
    tensor_i &ti = t;
    any_tensor<1, int> tt1(ti);
    any_tensor<2, double> tt2(ti);

    iface::tensor_list tl;
    size_t tid1 = tl.get_tensor_id(tt1);
    size_t tid2 = tl.get_tensor_id(tt2);

    if(tl.get_tensor_order(tid1) != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order (1).");
    }
    if(tl.get_tensor_type(tid1) != typeid(int)) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor type (1).");
    }
    if(tl.get_tensor_order(tid2) != 2) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order (2).");
    }
    if(tl.get_tensor_type(tid2) != typeid(double)) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor type (2).");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void tensor_list_test::test_2() {

    static const char testname[] = "tensor_list_test::test_2()";

    try {

    bispace<1> o(10);
    bispace<2> oo(o&o);

    btensor<1> bt1(o);
    btensor<2> bt2(oo);

    any_tensor<1, double> &at1 = bt1;
    any_tensor<2, double> &at2 = bt2;

    iface::tensor_list tl;
    size_t tid1 = tl.get_tensor_id(at1);
    size_t tid2 = tl.get_tensor_id(at2);

    if(tl.get_tensor_order(tid1) != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order (1).");
    }
    if(tl.get_tensor_type(tid1) != typeid(double)) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor type (1).");
    }
    if(tl.get_tensor_order(tid2) != 2) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order (2).");
    }
    if(tl.get_tensor_type(tid2) != typeid(double)) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor type (2).");
    }

    if(tl.get_tensor_id(bt1) != tid1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor id (1).");
    }
    if(tl.get_tensor_id(bt2) != tid2) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor id (2).");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void tensor_list_test::test_3() {

    static const char testname[] = "tensor_list_test::test_3()";

    try {

    bispace<1> o(10);
    bispace<2> oo(o&o);

    btensor<1> bt1(o);
    btensor<2> bt2(oo);

    any_tensor<1, double> &at1 = bt1;
    any_tensor<2, double> &at2 = bt2;

    iface::tensor_list tl;
    size_t tid1 = tl.get_tensor_id(bt1);
    size_t tid2 = tl.get_tensor_id(bt2);

    if(tl.get_tensor_order(tid1) != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order (1).");
    }
    if(tl.get_tensor_type(tid1) != typeid(double)) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor type (1).");
    }
    if(tl.get_tensor_order(tid2) != 2) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order (2).");
    }
    if(tl.get_tensor_type(tid2) != typeid(double)) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor type (2).");
    }

    if(tl.get_tensor_id(at1) != tid1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor id (1).");
    }
    if(tl.get_tensor_id(at2) != tid2) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor id (2).");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void tensor_list_test::test_4() {

    static const char testname[] = "tensor_list_test::test_4()";

    try {

    tensor t;
    tensor_i &ti = t;
    any_tensor<1, int> tt1(ti);
    any_tensor<2, double> tt2(ti);

    iface::tensor_list *tl1 = new iface::tensor_list;
    size_t tid1 = tl1->get_tensor_id(tt1);
    size_t tid2 = tl1->get_tensor_id(tt2);

    iface::tensor_list *tl2 = new iface::tensor_list(*tl1);
    delete tl1; tl1 = 0;

    any_tensor<1, int> &tt1a = tl2->get_tensor<1, int>(tid1);
    any_tensor<2, double> &tt2a = tl2->get_tensor<2, double>(tid2);

    delete tl2; tl2 = 0;

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void tensor_list_test::test_5() {

    static const char testname[] = "tensor_list_test::test_5()";

    try {

    tensor t;
    tensor_i &ti = t;
    any_tensor<1, int> tt1(ti);
    any_tensor<2, double> tt2(ti);

    iface::tensor_list *tl1 = new iface::tensor_list;
    size_t tid1 = tl1->get_tensor_id(tt1);
    size_t tid2 = tl1->get_tensor_id(tt2);

    iface::tensor_list *tl2 = new iface::tensor_list(*tl1, 1);
    delete tl1; tl1 = 0;

    any_tensor<1, int> &tt1a = tl2->get_tensor<1, int>(tid1);
    any_tensor<2, double> &tt2a = tl2->get_tensor<2, double>(tid2);

    delete tl2; tl2 = 0;

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void tensor_list_test::test_6() {

    static const char testname[] = "tensor_list_test::test_6()";

    try {

    tensor t;
    tensor_i &ti = t;
    any_tensor<1, int> tt1(ti);
    any_tensor<2, double> tt2(ti);
    any_tensor<3, int> tt3(ti);

    iface::tensor_list tl1, tl2;
    size_t tid1 = tl1.get_tensor_id(tt1);
    size_t tid2 = tl1.get_tensor_id(tt2);
    size_t tid3 = tl2.get_tensor_id(tt3);

    iface::tensor_list tl3(tl1);
    tl3.merge(tl2);

    any_tensor<1, int> &tt1a = tl3.get_tensor<1, int>(tid1);
    any_tensor<2, double> &tt2a = tl3.get_tensor<2, double>(tid2);
    any_tensor<3, int> &tt3a = tl3.get_tensor<3, int>(tid3);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    } catch(std::exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
