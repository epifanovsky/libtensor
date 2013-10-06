#include <algorithm>
#include <sstream>
#include <libtensor/exception.h>
#include <libtensor/expr/node_mult.h>
#include <libtensor/expr/node_ident.h>
#include "node_mult_test.h"

namespace libtensor {


void node_mult_test::perform() throw(libtest::test_exception) {

    test_1(false);
    test_1(true);
}


using namespace expr;


void node_mult_test::test_1(bool recip) throw(libtest::test_exception) {

    static const char testname[] = "node_mult_test::test_1()";

    try {

    node_ident t1(2), t2(3);
    node_mult c1(t1, t2, recip);

    node_mult *c1copy = c1.clone();
    if (c1copy->get_nargs() != 2) {
        fail_test(testname, __FILE__, __LINE__,
                "Wrong number of arguments.");
    }
    if (c1copy->do_recip() != recip) {
        fail_test(testname, __FILE__, __LINE__,
                "recip.");
    }

    std::vector<unsigned> ids(2); ids[0] = 2; ids[1] = 3;
    for (size_t i = 0; i < c1copy->get_nargs(); i++) {
        const node &ni = c1copy->get_arg(i);
        if (ni.get_op() != "ident") {
            std::ostringstream oss;
            oss << "Node type (node " << i << ").";
            fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
        }
        const node_ident &id = ni.recast_as<node_ident>();
        if (id.get_tid() != ids[i]) {
            std::ostringstream oss;
            oss << "Tensor ID (node " << i << ").";
            fail_test(testname, __FILE__, __LINE__, oss.str().c_str());
        }
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
