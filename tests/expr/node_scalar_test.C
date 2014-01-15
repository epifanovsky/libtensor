#include <memory>
#include <libtensor/exception.h>
#include <libtensor/expr/node_scalar.h>
#include "node_scalar_test.h"

namespace libtensor {


void node_scalar_test::perform() throw(libtest::test_exception) {

    test_1();
}


using namespace expr;


void node_scalar_test::test_1() {

    static const char testname[] = "node_scalar_test::test_1()";

    try {

    node_scalar<double> n1(0.2);

    if(n1.get_n() != 0) {
        fail_test(testname, __FILE__, __LINE__, "n1.get_n() != 0");
    }
    if(n1.get_type() != typeid(double)) {
        fail_test(testname, __FILE__, __LINE__,
            "n1.get_type() != typeid(double)");
    }
    if(n1.get_c() != 0.2) {
        fail_test(testname, __FILE__, __LINE__, "n1.get_c() != 0.2");
    }

    std::auto_ptr< node_scalar<double> > n1copy(
        dynamic_cast< node_scalar<double>* >(n1.clone()));

    if(n1copy->get_n() != 0) {
        fail_test(testname, __FILE__, __LINE__, "n1copy->get_n() != 0");
    }
    if(n1copy->get_type() != typeid(double)) {
        fail_test(testname, __FILE__, __LINE__,
            "n1copy->get_type() != typeid(double)");
    }
    if(n1copy->get_c() != 0.2) {
        fail_test(testname, __FILE__, __LINE__, "n1copy->get_c() != 0.2");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
