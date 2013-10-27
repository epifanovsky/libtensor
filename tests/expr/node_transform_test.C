#include <algorithm>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/exception.h>
#include <libtensor/expr/node_ident.h>
#include <libtensor/expr/node_transform.h>
#include "node_transform_test.h"

namespace libtensor {


void node_transform_test::perform() throw(libtest::test_exception) {

    test_1();
}


using namespace expr;


void node_transform_test::test_1() {

    static const char testname[] = "node_transform_test::test_1()";

    try {

    std::vector<size_t> order(4, 0);
    order[0] = 1; order[1] = 0; order[2] = 3; order[3] = 2;

    double c = 0.1;

    node_ident t1(2);
    node_transform<double> tr1(t1, order, scalar_transf<double>(c));

    node_transform<double> *tr1copy = tr1.clone();
    if (tr1copy->get_op() != "transform") {
        fail_test(testname, __FILE__, __LINE__, "Node name.");
    }
    if (tr1copy->get_type() != typeid(double)) {
        fail_test(testname, __FILE__, __LINE__, "Node transform type.");
    }
    if (tr1copy->get_perm().size() != 4) {
        fail_test(testname, __FILE__, __LINE__, "Length of index order.");
    }
    if (tr1copy->get_coeff().get_coeff() != 0.1) {
        fail_test(testname, __FILE__, __LINE__, "Coefficient.");
    }
    const node &n1 = tr1copy->get_arg();
    if (n1.get_op() != "ident") {
        fail_test(testname, __FILE__, __LINE__, "Argument type.");
    }
    const node_ident &i1 = n1.recast_as<node_ident>();
    if (i1.get_tid() != 2) {
        fail_test(testname, __FILE__, __LINE__, "Tensor ID (left).");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
