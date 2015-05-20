#include <memory>
#include <libtensor/exception.h>
#include <libtensor/expr/dag/node_set.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include "node_set_test.h"

namespace libtensor {


void node_set_test::perform() throw(libtest::test_exception) {

    test_1();
}


using namespace expr;


void node_set_test::test_1() throw(libtest::test_exception) {

    static const char testname[] = "node_set_test::test_1()";

    try {

    std::vector<size_t> idx(4);
    idx[0] = 0; idx[1] = 0; idx[2] = 1; idx[3] = 2;

    node_set s(idx);

    if(s.get_idx() != idx) {
        fail_test(testname, __FILE__, __LINE__, "Inconsistent tensor indices.");
    }

    std::auto_ptr<node_set> scopy(dynamic_cast<node_set*>(s.clone()));
    if(scopy->get_idx() != idx) {
        fail_test(testname, __FILE__, __LINE__, "Inconsistent tensor indices.");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
