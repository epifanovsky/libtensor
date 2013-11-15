#include <algorithm>
#include <libtensor/exception.h>
#include <libtensor/expr/node_diag.h>
#include <libtensor/expr/node_ident.h>
#include "node_diag_test.h"

namespace libtensor {


void node_diag_test::perform() throw(libtest::test_exception) {

    test_1();
}


using namespace expr;


void node_diag_test::test_1() throw(libtest::test_exception) {

    static const char testname[] = "node_diag_test::test_1()";

    try {

    std::vector<size_t> diag_dims(2);
    for (size_t i = 0; i < 2; i++) diag_dims[i] = i + 1;

    node_diag d1(1, diag_dims);

    node_diag *d1copy = d1.clone();
    if (d1copy->get_diag_dims() != diag_dims) {
        fail_test(testname, __FILE__, __LINE__,
                "Inconsistent diagonal dimensions.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
