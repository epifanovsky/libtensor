#include <algorithm>
#include <libtensor/exception.h>
#include <libtensor/expr/node_contract.h>
#include <libtensor/expr/node_ident.h>
#include "node_contract_test.h"

namespace libtensor {


void node_contract_test::perform() throw(libtest::test_exception) {

    test_1();
}


using namespace expr;


void node_contract_test::test_1() {

    static const char testname[] = "node_contract_test::test_1()";

    try {

    std::map<size_t, size_t> contr;
    contr[1] = 0;

    node_ident t1(2), t2(3);
    node_contract c1(t1, t2, contr);

    node_contract *c1copy = c1.clone();
    if (c1copy->get_contraction().size() != 1) {
        fail_test(testname, __FILE__, __LINE__,
                "Wrong length of contraction.");
    }

    const node &n1 = c1copy->get_left_arg();
    if (n1.get_op() != "ident") {
        fail_test(testname, __FILE__, __LINE__, "Node type (left).");
    }
    const node_ident &i1 = n1.recast_as<node_ident>();
    if (i1.get_tid() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Tensor ID (left).");
    }

    const node &n2 = c1copy->get_right_arg();
    if (n2.get_op() != "ident") {
        fail_test(testname, __FILE__, __LINE__, "Node type (right).");
    }
    const node_ident &i2 = n2.recast_as<node_ident>();
    if (i2.get_tid() != 3) {
        fail_test(testname, __FILE__, __LINE__, "Tensor ID (right).");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
