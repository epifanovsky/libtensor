#include <algorithm>
#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/expr/node_contract.h>
#include "node_contract_test.h"

namespace libtensor {


void node_contract_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
}


using namespace expr;


void node_contract_test::test_1() throw(libtest::test_exception) {

    static const char testname[] = "node_contract_test::test_1()";

    try {

    std::multimap<size_t, size_t> contr;
    contr.insert(std::pair<size_t, size_t>(1, 2));

    node_contract c1(3, contr, true);

    node_contract *c1copy = c1.clone();
    if (c1copy->get_op().compare(node_contract::k_op_type) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Wrong op type.");
    }
    if (c1copy->get_n() != 3) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order.");
    }
    if (c1copy->get_map() != contr) {
        fail_test(testname, __FILE__, __LINE__,
                "Inconsistent map.");
    }
    if (! c1copy->do_contract()) {
        fail_test(testname, __FILE__, __LINE__, "Wrong operation.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void node_contract_test::test_2() throw(libtest::test_exception) {

    static const char testname[] = "node_contract_test::test_2()";

    try {

    std::multimap<size_t, size_t> contr;
    contr.insert(std::pair<size_t, size_t>(1, 2));
    contr.insert(std::pair<size_t, size_t>(3, 4));

    node_contract c1(4, contr, true);

    node_contract *c1copy = c1.clone();
    if (c1copy->get_op().compare(node_contract::k_op_type) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Wrong op type.");
    }
    if (c1copy->get_n() != 4) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order.");
    }
    if (c1copy->get_map() != contr) {
        fail_test(testname, __FILE__, __LINE__,
                "Inconsistent map.");
    }
    if (! c1copy->do_contract()) {
        fail_test(testname, __FILE__, __LINE__, "Wrong operation.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
