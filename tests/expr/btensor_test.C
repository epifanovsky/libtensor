#include <cstdlib>
#include <sstream>
#include <libtensor/btod/btod_add.h>
#include <libtensor/btod/btod_random.h>
#include <libtensor/btensor/btensor.h>
#include <libtensor/expr/assignment_operator.h>
#include <libtensor/expr/operators.h>
#include "../compare_ref.h"
#include "btensor_test.h"

namespace libtensor {


void btensor_test::perform() throw(libtest::test_exception) {

    allocator<double>::vmm().init(16, 16, 16777216, 16777216);

    try {

    test_1();
    test_2();
    test_expr_copy_1();
    test_expr_copy_2();
    test_expr_copy_3();
    test_expr_copy_4();
    test_expr_add_1();
    test_expr_add_2();
    test_expr_add_3();
    test_expr_add_4();
    test_expr_add_5();

    } catch(...) {
        allocator<double>::vmm().shutdown();
        throw;
    }

    allocator<double>::vmm().shutdown();
}


/** \test Checks the dimensions of a new btensor
 **/
void btensor_test::test_1() throw(libtest::test_exception) {

    static const char *testname = "btensor_test::test_1()";

    try {

    bispace<1> i_sp(10), a_sp(20);
    i_sp.split(5); a_sp.split(5).split(10).split(15);
    bispace<2> ia(i_sp|a_sp);
    btensor<2> bt2(ia);

    dimensions<2> bt2_dims(bt2.get_bis().get_dims());
    if(bt2_dims[0] != 10) {
        fail_test("btensor_test::perform()", __FILE__, __LINE__,
            "Block tensor bt2 has the wrong dimension: i");
    }

    if(bt2_dims[1] != 20) {
        fail_test("btensor_test::perform()", __FILE__, __LINE__,
            "Block tensor bt2 has the wrong dimension: a");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Checks operator() with various letter labels
 **/
void btensor_test::test_2() throw(libtest::test_exception) {

    static const char *testname = "btensor_test::test_2()";

    try {

    bispace<1> s(10);
    bispace<2> ss(s&s);

    letter i, j;

    btensor<1> bt1(s);
    bt1(i);
    letter_expr<1> le_i(i);
    bt1(le_i);

    btensor<2> bt2(ss);
    bt2(i|j);
    letter_expr<2> le_ij(i|j);
    bt2(le_ij);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void btensor_test::test_expr_copy_1() throw(libtest::test_exception) {

    // b(i|j) = a(i|j)

    static const char *testname = "btensor_test::test_expr_copy_1()";

    try {

    bispace<1> sp_i(4), sp_j(4);
    bispace<2> sp_ij(sp_i&sp_j);
    btensor<2> bta(sp_ij), btb(sp_ij), btb_ref(sp_ij);

    block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
        btctrlb_ref(btb_ref);
    index<2> i0;
    dense_tensor_i<2, double> &ta = btctrla.req_block(i0);
    dense_tensor_i<2, double> &tb = btctrlb.req_block(i0);
    dense_tensor_i<2, double> &tb_ref = btctrlb_ref.req_block(i0);

    dimensions<2> dims(ta.get_dims());

    {
        dense_tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);
        double *dta = tca.req_dataptr();
        double *dtb1 = tcb.req_dataptr();
        double *dtb2 = tcb_ref.req_dataptr();

        // Fill in random data

        abs_index<2> aida(dims);
        do {
            size_t i = aida.get_abs_index();
            dta[i] = dtb2[i] = drand48();
            dtb1[i] = drand48();
        } while(aida.inc());

        tca.ret_dataptr(dta); dta = NULL;
        tcb.ret_dataptr(dtb1); dtb1 = NULL;
        tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
    }

    bta.set_immutable(); btb_ref.set_immutable();

    // Evaluate the expression

    letter i, j;
    btb(i|j) = bta(i|j);

    // Compare against the reference

    compare_ref<2>::compare(testname, btb, btb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void btensor_test::test_expr_copy_2() throw(libtest::test_exception) {

    // b(i|j) = a(j|i)

    static const char *testname = "btensor_test::test_expr_copy_2()";

    try {

    bispace<1> sp_i(4), sp_j(4);
    bispace<2> sp_ij(sp_i&sp_j);
    btensor<2> bta(sp_ij), btb(sp_ij), btb_ref(sp_ij);

    block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
        btctrlb_ref(btb_ref);
    index<2> i0;
    dense_tensor_i<2, double> &ta = btctrla.req_block(i0);
    dense_tensor_i<2, double> &tb = btctrlb.req_block(i0);
    dense_tensor_i<2, double> &tb_ref = btctrlb_ref.req_block(i0);

    dimensions<2> dims(ta.get_dims());

    {
        dense_tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);
        double *dta = tca.req_dataptr();
        double *dtb1 = tcb.req_dataptr();
        double *dtb2 = tcb_ref.req_dataptr();

        // Fill in random data

        abs_index<2> aida(dims);
        permutation<2> p; p.permute(0, 1);
        do {
            index<2> idb(aida.get_index());
            idb.permute(p);
            abs_index<2> aidb(idb, dims);
            size_t i = aida.get_abs_index();
            size_t j = aidb.get_abs_index();
            dta[i] = dtb2[j] = drand48();
            dtb1[j] = drand48();
        } while(aida.inc());

        tca.ret_dataptr(dta); dta = NULL;
        tcb.ret_dataptr(dtb1); dtb1 = NULL;
        tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
    }

    bta.set_immutable(); btb_ref.set_immutable();

    // Evaluate the expression

    letter i, j;
    btb(i|j) = bta(j|i);

    // Compare against the reference

    compare_ref<2>::compare(testname, btb, btb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void btensor_test::test_expr_copy_3() throw(libtest::test_exception) {

    // b(i|j) = 1.5*a(j|i)

    static const char *testname = "btensor_test::test_expr_copy_3()";

    try {

    bispace<1> sp_i(4), sp_j(4);
    bispace<2> sp_ij(sp_i&sp_j);
    btensor<2> bta(sp_ij), btb(sp_ij), btb_ref(sp_ij);

    block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
        btctrlb_ref(btb_ref);
    index<2> i0;
    dense_tensor_i<2, double> &ta = btctrla.req_block(i0);
    dense_tensor_i<2, double> &tb = btctrlb.req_block(i0);
    dense_tensor_i<2, double> &tb_ref = btctrlb_ref.req_block(i0);

    dimensions<2> dims(ta.get_dims());

    {
        dense_tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);
        double *dta = tca.req_dataptr();
        double *dtb1 = tcb.req_dataptr();
        double *dtb2 = tcb_ref.req_dataptr();

        // Fill in random data

        abs_index<2> aida(dims);
        permutation<2> p; p.permute(0, 1);
        do {
            index<2> idb(aida.get_index());
            idb.permute(p);
            abs_index<2> aidb(idb, dims);
            size_t i = aida.get_abs_index();
            size_t j = aidb.get_abs_index();
            dta[i] = drand48();
            dtb2[j] = 1.5 * dta[i];
            dtb1[j] = drand48();
        } while(aida.inc());

        tca.ret_dataptr(dta); dta = NULL;
        tcb.ret_dataptr(dtb1); dtb1 = NULL;
        tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
    }

    bta.set_immutable(); btb_ref.set_immutable();

    // Evaluate the expression

    letter i, j;
    btb(i|j) = 1.5 * bta(j|i);

    // Compare against the reference

    compare_ref<2>::compare(testname, btb, btb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void btensor_test::test_expr_copy_4() throw(libtest::test_exception) {

    // b(i|j) = -a(i|j)

    static const char *testname = "btensor_test::test_expr_copy_4()";

    try {

    bispace<1> sp_i(4), sp_j(4);
    bispace<2> sp_ij(sp_i&sp_j);
    btensor<2> bta(sp_ij), btb(sp_ij), btb_ref(sp_ij);

    block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
        btctrlb_ref(btb_ref);
    index<2> i0;
    dense_tensor_i<2, double> &ta = btctrla.req_block(i0);
    dense_tensor_i<2, double> &tb = btctrlb.req_block(i0);
    dense_tensor_i<2, double> &tb_ref = btctrlb_ref.req_block(i0);

    dimensions<2> dims(ta.get_dims());

    {
        dense_tensor_ctrl<2, double> tca(ta), tcb(tb), tcb_ref(tb_ref);
        double *dta = tca.req_dataptr();
        double *dtb1 = tcb.req_dataptr();
        double *dtb2 = tcb_ref.req_dataptr();

        // Fill in random data

        abs_index<2> aida(dims);
        do {
            size_t i = aida.get_abs_index();
            dta[i] = drand48();
            dtb2[i] = -dta[i];
            dtb1[i] = drand48();
        } while(aida.inc());

        tca.ret_dataptr(dta); dta = NULL;
        tcb.ret_dataptr(dtb1); dtb1 = NULL;
        tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
    }

    bta.set_immutable(); btb_ref.set_immutable();

    // Evaluate the expression

    letter i, j;
    btb(i|j) = -bta(i|j);

    // Compare against the reference

    compare_ref<2>::compare(testname, btb, btb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void btensor_test::test_expr_add_1() throw(libtest::test_exception) {

    // c(i|j) = a(i|j) + b(i|j)

    static const char *testname = "btensor_test::test_expr_add_1()";

    try {

    bispace<1> sp_i(4), sp_j(4);
    bispace<2> sp_ij(sp_i&sp_j);
    btensor<2> bta(sp_ij), btb(sp_ij), btc(sp_ij), btc_ref(sp_ij);

    block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb),
        btctrlc(btc), btctrlc_ref(btc_ref);
    index<2> i0;
    dense_tensor_i<2, double> &ta = btctrla.req_block(i0);
    dense_tensor_i<2, double> &tb = btctrlb.req_block(i0);
    dense_tensor_i<2, double> &tc = btctrlc.req_block(i0);
    dense_tensor_i<2, double> &tc_ref = btctrlc_ref.req_block(i0);

    dimensions<2> dims(ta.get_dims());

    {
        dense_tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);
        double *dta = tca.req_dataptr();
        double *dtb = tcb.req_dataptr();
        double *dtc1 = tcc.req_dataptr();
        double *dtc2 = tcc_ref.req_dataptr();

        // Fill in random data

        abs_index<2> aida(dims);
        do {
            size_t i = aida.get_abs_index();
            dta[i] = drand48();
            dtb[i] = drand48();
            dtc1[i] = drand48();
            dtc2[i] = dta[i] + dtb[i];
        } while(aida.inc());

        tca.ret_dataptr(dta); dta = NULL;
        tcb.ret_dataptr(dtb); dtb = NULL;
        tcc.ret_dataptr(dtc1); dtc1 = NULL;
        tcc_ref.ret_dataptr(dtc2); dtc2 = NULL;
    }

    bta.set_immutable();
    btb.set_immutable();
    btc_ref.set_immutable();

    // Evaluate the expression

    letter i, j;
    btc(i|j) = bta(i|j) + btb(i|j);

    // Compare against the reference

    compare_ref<2>::compare(testname, btc, btc_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void btensor_test::test_expr_add_2() throw(libtest::test_exception) {

    // c(i|j) = -a(i|j) + 3.0*b(i|j)

    static const char *testname = "btensor_test::test_expr_add_2()";

    try {

    bispace<1> sp_i(4), sp_j(4);
    bispace<2> sp_ij(sp_i&sp_j);
    btensor<2> bta(sp_ij), btb(sp_ij), btc(sp_ij), btc_ref(sp_ij);

    block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb), btctrlc(btc),
        btctrlc_ref(btc_ref);
    index<2> i0;
    dense_tensor_i<2, double> &ta = btctrla.req_block(i0);
    dense_tensor_i<2, double> &tb = btctrlb.req_block(i0);
    dense_tensor_i<2, double> &tc = btctrlc.req_block(i0);
    dense_tensor_i<2, double> &tc_ref = btctrlc_ref.req_block(i0);

    dimensions<2> dims(ta.get_dims());

    {
        dense_tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);
        double *dta = tca.req_dataptr();
        double *dtb = tcb.req_dataptr();
        double *dtc1 = tcc.req_dataptr();
        double *dtc2 = tcc_ref.req_dataptr();

        // Fill in random data

        abs_index<2> aida(dims);
        do {
            size_t i = aida.get_abs_index();
            dta[i] = drand48();
            dtb[i] = drand48();
            dtc1[i] = drand48();
            dtc2[i] = -dta[i] + 3.0*dtb[i];
        } while(aida.inc());

        tca.ret_dataptr(dta); dta = NULL;
        tcb.ret_dataptr(dtb); dtb = NULL;
        tcc.ret_dataptr(dtc1); dtc1 = NULL;
        tcc_ref.ret_dataptr(dtc2); dtc2 = NULL;
    }

    bta.set_immutable();
    btb.set_immutable();
    btc_ref.set_immutable();

    // Evaluate the expression

    letter i, j;
    btc(i|j) = -bta(i|j) + 3.0*btb(i|j);

    // Compare against the reference

    compare_ref<2>::compare(testname, btc, btc_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void btensor_test::test_expr_add_3() throw(libtest::test_exception) {

    // c(i|j) = a(i|j) - b(i|j)

    static const char *testname = "btensor_test::test_expr_add_3()";

    try {

    bispace<1> sp_i(4), sp_j(4);
    bispace<2> sp_ij(sp_i&sp_j);
    btensor<2> bta(sp_ij), btb(sp_ij), btc(sp_ij), btc_ref(sp_ij);

    block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb), btctrlc(btc),
        btctrlc_ref(btc_ref);
    index<2> i0;
    dense_tensor_i<2, double> &ta = btctrla.req_block(i0);
    dense_tensor_i<2, double> &tb = btctrlb.req_block(i0);
    dense_tensor_i<2, double> &tc = btctrlc.req_block(i0);
    dense_tensor_i<2, double> &tc_ref = btctrlc_ref.req_block(i0);

    dimensions<2> dims(ta.get_dims());

    {
        dense_tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);
        double *dta = tca.req_dataptr();
        double *dtb = tcb.req_dataptr();
        double *dtc1 = tcc.req_dataptr();
        double *dtc2 = tcc_ref.req_dataptr();

        // Fill in random data

        abs_index<2> aida(dims);
        do {
            size_t i = aida.get_abs_index();
            dta[i] = drand48();
            dtb[i] = drand48();
            dtc1[i] = drand48();
            dtc2[i] = dta[i] - dtb[i];
        } while(aida.inc());

        tca.ret_dataptr(dta); dta = NULL;
        tcb.ret_dataptr(dtb); dtb = NULL;
        tcc.ret_dataptr(dtc1); dtc1 = NULL;
        tcc_ref.ret_dataptr(dtc2); dtc2 = NULL;
    }

    bta.set_immutable();
    btb.set_immutable();
    btc_ref.set_immutable();

    // Evaluate the expression

    letter i, j;
    btc(i|j) = bta(i|j) - btb(i|j);

    // Compare against the reference

    compare_ref<2>::compare(testname, btc, btc_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void btensor_test::test_expr_add_4() throw(libtest::test_exception) {

    // c(i|j) = 4.0*a(i|j) - 0.5*b(j|i)

    static const char *testname = "btensor_test::test_expr_add_4()";

    try {

    bispace<1> sp_i(4), sp_j(4);
    bispace<2> sp_ij(sp_i&sp_j);
    btensor<2> bta(sp_ij), btb(sp_ij), btc(sp_ij), btc_ref(sp_ij);

    block_tensor_ctrl<2, double> btctrla(bta), btctrlb(btb), btctrlc(btc),
        btctrlc_ref(btc_ref);
    index<2> i0;
    dense_tensor_i<2, double> &ta = btctrla.req_block(i0);
    dense_tensor_i<2, double> &tb = btctrlb.req_block(i0);
    dense_tensor_i<2, double> &tc = btctrlc.req_block(i0);
    dense_tensor_i<2, double> &tc_ref = btctrlc_ref.req_block(i0);

    dimensions<2> dims(ta.get_dims());

    {
        dense_tensor_ctrl<2, double> tca(ta), tcb(tb), tcc(tc), tcc_ref(tc_ref);
        double *dta = tca.req_dataptr();
        double *dtb = tcb.req_dataptr();
        double *dtc1 = tcc.req_dataptr();
        double *dtc2 = tcc_ref.req_dataptr();

        // Fill in random data

        abs_index<2> aida(dims);
        permutation<2> p; p.permute(0, 1);
        do {
            index<2> idb(aida.get_index());
            idb.permute(p);
            abs_index<2> aidb(idb, dims);
            size_t i = aida.get_abs_index();
            size_t j = aidb.get_abs_index();
            dta[i] = drand48();
            dtb[j] = drand48();
            dtc1[i] = drand48();
            dtc2[i] = 4.0*dta[i] - 0.5*dtb[j];
        } while(aida.inc());

        tca.ret_dataptr(dta); dta = NULL;
        tcb.ret_dataptr(dtb); dtb = NULL;
        tcc.ret_dataptr(dtc1); dtc1 = NULL;
        tcc_ref.ret_dataptr(dtc2); dtc2 = NULL;
    }

    bta.set_immutable();
    btb.set_immutable();
    btc_ref.set_immutable();

    // Evaluate the expression

    letter i, j;
    btc(i|j) = 4.0*bta(i|j) - 0.5*btb(j|i);

    // Compare against the reference

    compare_ref<2>::compare(testname, btc, btc_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void btensor_test::test_expr_add_5() throw(libtest::test_exception) {

    // d(i|j) = a(i|j) + b(i|j) + c(i|j)

    static const char *testname = "btensor_test::test_expr_add_5()";

    try {

    bispace<1> sp_i(4), sp_j(4);
    bispace<2> sp_ij(sp_i&sp_j);
    btensor<2> bta(sp_ij), btb(sp_ij), btc(sp_ij), btd(sp_ij), btd_ref(sp_ij);

    //      Fill in random data

    btod_random<2>().perform(bta);
    btod_random<2>().perform(btb);
    btod_random<2>().perform(btc);

    //      Compute the reference

    btod_add<2> add(bta);
    add.add_op(btb);
    add.add_op(btc);
    add.perform(btd_ref);

    //      Evaluate the expression

    letter i, j;
    btd(i|j) = bta(i|j) + btb(i|j) + btc(i|j);

    //      Compare against the reference

    compare_ref<2>::compare(testname, btd, btd_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


} // namespace libtensor

