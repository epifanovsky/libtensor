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

    node_ident n1(0);
    plan.add_node(node_assign(1, n1));

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
