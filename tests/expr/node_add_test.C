#include <algorithm>
#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/expr/dag/node_add.h>
#include "node_add_test.h"

namespace libtensor {


void node_add_test::perform() throw(libtest::test_exception) {

    test_1();
}


using namespace expr;


void node_add_test::test_1() throw(libtest::test_exception) {

    static const char testname[] = "node_add_test::test_1()";

    try {

    node_add c1(5);
    node_add *c1copy = dynamic_cast<node_add*>(c1.clone());
    if (c1copy->get_op().compare(node_add::k_op_type) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Wrong op type.");
    }
    if (c1copy->get_n() != 5) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
