#include <memory>
#include <libtensor/exception.h>
#include <libtensor/expr/dag/node_trace.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include "node_trace_test.h"

namespace libtensor {


void node_trace_test::perform() throw(libtest::test_exception) {

    test_1();
    test_2();
}


using namespace expr;


void node_trace_test::test_1() {

    static const char testname[] = "node_trace_test::test_1()";

    try {

    std::vector<size_t> idx(2), cidx(2);
    idx[0] = 0; idx[1] = 1;
    cidx[0] = 0; cidx[1] = 1;

    node_trace d1(idx, cidx);

    if(d1.get_n() != 0) {
        fail_test(testname, __FILE__, __LINE__, "d1.get_n() != 0");
    }
    if(d1.get_idx() != idx) {
        fail_test(testname, __FILE__, __LINE__, "d1.get_idx() != idx");
    }
    if(d1.get_cidx() != cidx) {
        fail_test(testname, __FILE__, __LINE__, "d1.get_cidx() != cidx");
    }

    std::auto_ptr<node_trace> d1copy(dynamic_cast<node_trace*>(d1.clone()));
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


void node_trace_test::test_2() {

    static const char testname[] = "node_trace_test::test_2()";

    try {

    std::vector<size_t> idx(4), cidx(2);
    idx[0] = 0; idx[1] = 1; idx[2] = 0; idx[3] = 1;
    cidx[0] = 0; cidx[1] = 1;

    node_trace d1(idx, cidx);

    if(d1.get_n() != 0) {
        fail_test(testname, __FILE__, __LINE__, "d1.get_n() != 0");
    }
    if(d1.get_idx() != idx) {
        fail_test(testname, __FILE__, __LINE__, "d1.get_idx() != idx");
    }
    if(d1.get_cidx() != cidx) {
        fail_test(testname, __FILE__, __LINE__, "d1.get_cidx() != cidx");
    }

    std::auto_ptr<node_trace> d1copy(dynamic_cast<node_trace*>(d1.clone()));
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
