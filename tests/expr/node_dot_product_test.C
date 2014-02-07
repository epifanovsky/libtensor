#include <memory>
#include <libtensor/exception.h>
#include <libtensor/expr/node_dot_product.h>
#include <libtensor/expr/node_ident_any_tensor.h>
#include "node_dot_product_test.h"

namespace libtensor {


void node_dot_product_test::perform() throw(libtest::test_exception) {

    test_1();
}


using namespace expr;


void node_dot_product_test::test_1() {

    static const char testname[] = "node_dot_product_test::test_1()";

    try {

    std::vector<size_t> idxa(2), idxb(2), idx(4), cidx(2);
    idxa[0] = 0; idxa[1] = 1;
    idxb[0] = 1; idxb[1] = 0;
    idx[0] = 0; idx[1] = 1; idx[2] = 1; idx[3] = 0;
    cidx[0] = 0; cidx[1] = 1;

    node_dot_product d1(idxa, idxb);

    if(d1.get_n() != 0) {
        fail_test(testname, __FILE__, __LINE__, "d1.get_n() != 0");
    }
    if(d1.get_idx() != idx) {
        fail_test(testname, __FILE__, __LINE__, "d1.get_idx() != idx");
    }
    if(d1.get_cidx() != cidx) {
        fail_test(testname, __FILE__, __LINE__, "d1.get_cidx() != cidx");
    }

    std::auto_ptr<node_dot_product> d1copy(
        dynamic_cast<node_dot_product*>(d1.clone()));
    if(d1copy->get_idx() != idx) {
        fail_test(testname, __FILE__, __LINE__, "d1copy->get_idx() != idx");
    }
    if(d1copy->get_cidx() != cidx) {
        fail_test(testname, __FILE__, __LINE__, "d1copy->get_cidx() != cidx");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
