#include <algorithm>
#include <libtensor/exception.h>
#include <libtensor/expr/eval_plan.h>
#include <libtensor/expr/node_ident.h>
#include "eval_plan_test.h"

namespace libtensor {


void eval_plan_test::perform() throw(libtest::test_exception) {

    test_1();
}


using namespace expr;


void eval_plan_test::test_1() {

    static const char testname[] = "eval_plan_test::test_1()";

    try {

    eval_plan plan;

    node_ident n1(2, 5);
    plan.insert_assignment(node_assign(1, n1));

    if(std::distance(plan.begin(), plan.end()) != 1) {
        fail_test(testname, __FILE__, __LINE__, "Wrong length of list.");
    }

    for(eval_plan::iterator i = plan.begin(); i != plan.end(); ++i) {

        const eval_plan_item &item = plan.get_item(i);
        if(item.code != eval_plan_action_code::ASSIGN) {
            fail_test(testname, __FILE__, __LINE__, "Wrong action code.");
        }
        if(item.node->get_tid() != 1) {
            fail_test(testname, __FILE__, __LINE__, "Wrong LHS tensor ID.");
        }
        const node_ident &n = item.node->get_rhs().recast_as<node_ident>();
        if(n.get_tid() != 5) {
            fail_test(testname, __FILE__, __LINE__, "Wrong RHS tensor ID.");
        }
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
