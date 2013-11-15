#include <algorithm>
#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/expr/node_add.h>
#include "node_add_test.h"

namespace libtensor {


void node_add_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
}


using namespace expr;


void node_add_test::test_1() throw(libtest::test_exception) {

    static const char testname[] = "node_add_test::test_1()";

    try {


    std::multimap<size_t, size_t> map;
    map.insert(std::pair<size_t, size_t>(0, 3));
    map.insert(std::pair<size_t, size_t>(1, 2));

    node_add c1(5, map);

    node_add *c1copy = c1.clone();
    if (c1copy->get_op().compare(node_add::k_op_type) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Wrong op type.");
    }
    if (c1copy->get_n() != 5) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order.");
    }

    if (c1copy->get_map() != map) {
        fail_test(testname, __FILE__, __LINE__, "Inconsistent map.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void node_add_test::test_2() throw(libtest::test_exception) {

    static const char testname[] = "node_add_test::test_2()";

    try {

    std::multimap<size_t, size_t> map;
    node_add c1(4, map);

    node_add *c1copy = c1.clone();
    if (c1copy->get_op().compare(node_add::k_op_type) != 0) {
        fail_test(testname, __FILE__, __LINE__, "Wrong op type.");
    }
    if (c1copy->get_n() != 4) {
        fail_test(testname, __FILE__, __LINE__, "Wrong tensor order.");
    }

    if (c1copy->get_map() != map) {
        fail_test(testname, __FILE__, __LINE__, "Inconsistent map.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
