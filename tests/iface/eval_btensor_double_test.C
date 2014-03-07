#include <libtensor/block_tensor/btod_contract2.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/expr/dag/node_contract.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/expr/btensor/btensor.h>
#include <libtensor/expr/btensor/eval_btensor.h>
#include "../compare_ref.h"
#include "eval_btensor_double_test.h"

namespace libtensor {
using namespace expr;


void eval_btensor_double_test::perform() throw(libtest::test_exception) {

    allocator<double>::init(16, 16, 16777216, 16777216);

    try {

        test_copy_1();
        test_copy_2();
        test_copy_3();
        test_contract_1();

    } catch(...) {
        allocator<double>::shutdown();
        throw;
    }

    allocator<double>::shutdown();
}


void eval_btensor_double_test::test_copy_1() {

    static const char testname[] = "eval_btensor_double_test::test_copy_1()";

    try {

    bispace<1> o(10), v(20);
    bispace<2> oo(o&o), ov(o|v), vv(v&v);

    btensor<2, double> t_oo(oo), t_ov(ov), t_vv(vv);
    btensor<2, double> r_oo(oo), r_ov(ov), r_vv(vv);

    btod_random<2>().perform(t_oo);
    btod_random<2>().perform(t_ov);
    btod_random<2>().perform(t_vv);

//    tensor_list tl;

//    unsigned tid_oo = tl.get_tensor_id(t_oo);
//    unsigned tid_ov = tl.get_tensor_id(t_ov);
//    unsigned tid_vv = tl.get_tensor_id(t_vv);
//    unsigned rid_oo = tl.get_tensor_id(r_oo);
//    unsigned rid_ov = tl.get_tensor_id(r_ov);
//    unsigned rid_vv = tl.get_tensor_id(r_vv);

//    eval_plan plan;

    node_ident_any_tensor<2, double> nid1(t_oo), nid2(t_ov), nid3(t_vv);
//    plan.insert_assignment(node_assign(rid_oo, nid1));
//    plan.insert_assignment(node_assign(rid_ov, nid2));
//    plan.insert_assignment(node_assign(rid_vv, nid3));

//    eval_btensor<double>().process_plan(plan, tl);
//
//    compare_ref<2>::compare(testname, r_oo, t_oo, 1e-15);
//    compare_ref<2>::compare(testname, r_ov, t_ov, 1e-15);
//    compare_ref<2>::compare(testname, r_vv, t_vv, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void eval_btensor_double_test::test_copy_2() {

    static const char testname[] = "eval_btensor_double_test::test_copy_2()";

    try {

    bispace<1> o(10), v(20);
    bispace<2> oo(o&o), ov(o|v), vv(v&v);

    btensor<2, double> t_oo(oo), t_ov(ov), t_vv(vv);
    btensor<2, double> r_oo(oo), r_ov(ov), r_vv(vv);
    btensor<2, double> r_oo_ref(oo), r_ov_ref(ov), r_vv_ref(vv);

    btod_random<2>().perform(t_oo);
    btod_random<2>().perform(t_ov);
    btod_random<2>().perform(t_vv);

    btod_copy<2>(t_oo, -2.0).perform(r_oo_ref);
    btod_copy<2>(t_ov).perform(r_ov_ref);
    btod_copy<2>(t_vv, permutation<2>().permute(0, 1), 1.5).perform(r_vv_ref);

//    tensor_list tl;

//    unsigned tid_oo = tl.get_tensor_id(t_oo);
//    unsigned tid_ov = tl.get_tensor_id(t_ov);
//    unsigned tid_vv = tl.get_tensor_id(t_vv);
//    unsigned rid_oo = tl.get_tensor_id(r_oo);
//    unsigned rid_ov = tl.get_tensor_id(r_ov);
//    unsigned rid_vv = tl.get_tensor_id(r_vv);
//
//    eval_plan plan;

    std::vector<size_t> p01(2, 0), p10(2, 0);
    p01[0] = 0; p01[1] = 1;
    p10[0] = 1; p10[1] = 0;

    node_ident_any_tensor<2, double> nid1(t_oo), nid2(t_ov), nid3(t_vv);
    node_transform<double> ntr1(p01, scalar_transf<double>(-2.0)),
        ntr2(p01, scalar_transf<double>(1.0)),
        ntr3(p10, scalar_transf<double>(1.5));
//    plan.insert_assignment(node_assign(rid_oo, ntr1));
//    plan.insert_assignment(node_assign(rid_ov, ntr2));
//    plan.insert_assignment(node_assign(rid_vv, ntr3));

//    eval_btensor<double>().process_plan(plan, tl);
//
//    compare_ref<2>::compare(testname, r_oo, r_oo_ref, 1e-15);
//    compare_ref<2>::compare(testname, r_ov, r_ov_ref, 1e-15);
//    compare_ref<2>::compare(testname, r_vv, r_vv_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void eval_btensor_double_test::test_copy_3() {

    static const char testname[] = "eval_btensor_double_test::test_copy_3()";

    try {

    bispace<1> o(10);
    bispace<3> ooo(o&o&o);

    btensor<3, double> t_ooo(ooo);
    btensor<3, double> r_ooo(ooo);
    btensor<3, double> r_ooo_ref(ooo);

    btod_random<3>().perform(t_ooo);

    btod_copy<3>(t_ooo, permutation<3>().permute(0, 1).permute(1, 2)).
        perform(r_ooo_ref);

//    tensor_list tl;
//
//    unsigned tid_ooo = tl.get_tensor_id(t_ooo);
//    unsigned rid_ooo = tl.get_tensor_id(r_ooo);
//
//    eval_plan plan;

    std::vector<size_t> p012(3, 0), p102(3, 0), p021(3, 0);
    p012[0] = 0; p012[1] = 1; p012[2] = 2;
    p102[0] = 1; p102[1] = 0; p102[2] = 2;
    p021[0] = 0; p021[1] = 2; p021[2] = 1;

    node_ident_any_tensor<3, double> nid(t_ooo);
    node_transform<double> ntr1(p102, scalar_transf<double>(1.0)),
        ntr2(p021, scalar_transf<double>(1.0));
//    plan.insert_assignment(node_assign(rid_ooo, ntr2));

//    eval_btensor<double>().process_plan(plan, tl);
//
//    compare_ref<3>::compare(testname, r_ooo, r_ooo_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void eval_btensor_double_test::test_contract_1() {

    static const char testname[] =
        "eval_btensor_double_test::test_contract_1()";

    try {

    bispace<1> o(10), v(20);
    bispace<2> oo(o&o), ov(o|v), vv(v&v);

    btensor<2, double> t_oo(oo), t_ov(ov), t_vv(vv);
    btensor<2, double> r1_ov(ov), r2_ov(ov);
    btensor<2, double> r1_ov_ref(ov), r2_ov_ref(ov);

    btod_random<2>().perform(t_oo);
    btod_random<2>().perform(t_ov);
    btod_random<2>().perform(t_vv);

    {
        contraction2<1, 1, 1> contr1, contr2;
        contr1.contract(0, 0);
        contr2.contract(1, 1);
        btod_contract2<1, 1, 1>(contr1, t_oo, t_ov).perform(r1_ov_ref);
        btod_contract2<1, 1, 1>(contr2, t_ov, t_vv).perform(r2_ov_ref);
    }

    node_ident_any_tensor<2, double> nid1(t_oo), nid2(t_ov), nid3(t_vv);
    std::multimap<size_t, size_t> contr1, contr2;
    contr1.insert(std::pair<size_t, size_t>(0, 2));
    contr1.insert(std::pair<size_t, size_t>(1, 3));
    node_contract nco1(2, contr1, true), nco2(4, contr2, true);
//    plan.insert_assignment(node_assign(rid1_ov, nco1));
//    plan.insert_assignment(node_assign(rid2_ov, nco2));

//    eval_btensor<double>().process_plan(plan, tl);
//
//    compare_ref<2>::compare(testname, r1_ov, r1_ov_ref, 1e-15);
//    compare_ref<2>::compare(testname, r2_ov, r2_ov_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

