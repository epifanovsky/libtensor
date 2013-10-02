#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/expr/eval_plan.h>
#include <libtensor/expr/node_ident.h>
#include <libtensor/iface/btensor.h>
#include <libtensor/iface/btensor/eval_btensor.h>
#include "../compare_ref.h"
#include "eval_btensor_double_test.h"

namespace libtensor {
using namespace expr;
using namespace iface;


void eval_btensor_double_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);

    try {

        test_1();
        test_2();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void eval_btensor_double_test::test_1() {

    static const char testname[] = "eval_btensor_double_test::test_1()";

    try {

    bispace<1> o(10), v(20);
    bispace<2> oo(o&o), ov(o|v), vv(v&v);

    btensor<2, double> t_oo(oo), t_ov(ov), t_vv(vv);
    btensor<2, double> r_oo(oo), r_ov(ov), r_vv(vv);

    btod_random<2>().perform(t_oo);
    btod_random<2>().perform(t_ov);
    btod_random<2>().perform(t_vv);

    tensor_list tl;

    unsigned tid_oo = tl.get_tensor_id(t_oo);
    unsigned tid_ov = tl.get_tensor_id(t_ov);
    unsigned tid_vv = tl.get_tensor_id(t_vv);
    unsigned rid_oo = tl.get_tensor_id(r_oo);
    unsigned rid_ov = tl.get_tensor_id(r_ov);
    unsigned rid_vv = tl.get_tensor_id(r_vv);

    eval_plan plan;

    node_ident nid1(tid_oo), nid2(tid_ov), nid3(tid_vv);
    plan.insert_assignment(node_assign(rid_oo, nid1));
    plan.insert_assignment(node_assign(rid_ov, nid2));
    plan.insert_assignment(node_assign(rid_vv, nid3));

    eval_btensor<double>().process_plan(plan, tl);

    compare_ref<2>::compare(testname, r_oo, t_oo, 1e-15);
    compare_ref<2>::compare(testname, r_ov, t_ov, 1e-15);
    compare_ref<2>::compare(testname, r_vv, t_vv, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void eval_btensor_double_test::test_2() {

    static const char testname[] = "eval_btensor_double_test::test_2()";

    try {


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

