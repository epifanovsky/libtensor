#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/tod_copy.h>
#include "../compare_ref.h"
#include "tod_copy_test.h"

namespace libtensor {

typedef std_allocator<double> allocator_t;
typedef dense_tensor<4, double, allocator_t> tensor4;
typedef dense_tensor_ctrl<4, double> tensor4_ctrl;


void tod_copy_test::perform() throw (libtest::test_exception) {

    test_exc();

    index<2> i2a, i2b;
    i2b[0] = 10;
    i2b[1] = 12;
    index_range<2> ir2(i2a, i2b);
    dimensions<2> dims2(ir2);
    permutation<2> perm2, perm2t;
    perm2t.permute(0, 1);

    test_plain(dims2);
    test_plain_additive(dims2, 1.0);
    test_plain_additive(dims2, -1.0);
    test_plain_additive(dims2, 2.5);

    test_scaled(dims2, 1.0);
    test_scaled(dims2, 0.5);
    test_scaled(dims2, -3.14);
    test_scaled_additive(dims2, 1.0, 1.0);
    test_scaled_additive(dims2, 0.5, 1.0);
    test_scaled_additive(dims2, -3.14, 1.0);
    test_scaled_additive(dims2, 1.0, -1.0);
    test_scaled_additive(dims2, 0.5, -1.0);
    test_scaled_additive(dims2, -3.14, -1.0);
    test_scaled_additive(dims2, 1.0, 2.5);
    test_scaled_additive(dims2, 0.5, 2.5);
    test_scaled_additive(dims2, -3.14, 2.5);

    test_perm(dims2, perm2);
    test_perm(dims2, perm2t);
    test_perm_additive(dims2, perm2, 1.0);
    test_perm_additive(dims2, perm2, -1.0);
    test_perm_additive(dims2, perm2, 2.5);
    test_perm_additive(dims2, perm2t, 1.0);
    test_perm_additive(dims2, perm2t, -1.0);
    test_perm_additive(dims2, perm2t, 2.5);

    test_perm_scaled(dims2, perm2, 1.0);
    test_perm_scaled(dims2, perm2t, 1.0);
    test_perm_scaled(dims2, perm2, 0.5);
    test_perm_scaled(dims2, perm2t, 0.5);
    test_perm_scaled(dims2, perm2, -3.14);
    test_perm_scaled(dims2, perm2t, -3.14);
    test_perm_scaled_additive(dims2, perm2, 1.0, 1.0);
    test_perm_scaled_additive(dims2, perm2t, 1.0, 1.0);
    test_perm_scaled_additive(dims2, perm2, 0.5, 1.0);
    test_perm_scaled_additive(dims2, perm2t, 0.5, 1.0);
    test_perm_scaled_additive(dims2, perm2, -3.14, 1.0);
    test_perm_scaled_additive(dims2, perm2t, -3.14, 1.0);
    test_perm_scaled_additive(dims2, perm2, 1.0, -1.0);
    test_perm_scaled_additive(dims2, perm2t, 1.0, -1.0);
    test_perm_scaled_additive(dims2, perm2, 0.5, -1.0);
    test_perm_scaled_additive(dims2, perm2t, 0.5, -1.0);
    test_perm_scaled_additive(dims2, perm2, -3.14, -1.0);
    test_perm_scaled_additive(dims2, perm2t, -3.14, -1.0);
    test_perm_scaled_additive(dims2, perm2, 1.0, 2.5);
    test_perm_scaled_additive(dims2, perm2t, 1.0, 2.5);
    test_perm_scaled_additive(dims2, perm2, 0.5, 2.5);
    test_perm_scaled_additive(dims2, perm2t, 0.5, 2.5);
    test_perm_scaled_additive(dims2, perm2, -3.14, 2.5);
    test_perm_scaled_additive(dims2, perm2t, -3.14, 2.5);

    index<4> i4a, i4b;
    i4b[0] = 4;
    i4b[1] = 5;
    i4b[2] = 6;
    i4b[3] = 7;
    dimensions<4> dims4(index_range<4> (i4a, i4b));
    permutation<4> perm4, perm4c;
    perm4c.permute(0, 1).permute(1, 2).permute(2, 3);

    test_perm(dims4, perm4);
    test_perm(dims4, perm4c);

}


template<size_t N>
void tod_copy_test::test_plain(const dimensions<N> &dims)
    throw (libtest::test_exception) {

    static const char *testname = "tod_copy_test::test_plain()";

    try {

        dense_tensor<N, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);

        {
            dense_tensor_ctrl<N, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

            double *dta = tca.req_dataptr();
            double *dtb1 = tcb.req_dataptr();
            double *dtb2 = tcb_ref.req_dataptr();

            // Fill in random data

            abs_index<N> aida(dims);
            do {
                size_t i = aida.get_abs_index();
                dta[i] = dtb2[i] = drand48();
                dtb1[i] = drand48();
            } while(aida.inc());
            tca.ret_dataptr(dta);
            dta = NULL;
            tcb.ret_dataptr(dtb1);
            dtb1 = NULL;
            tcb_ref.ret_dataptr(dtb2);
            dtb2 = NULL;
            ta.set_immutable();
            tb_ref.set_immutable();
        }

        // Invoke the copy operation

        tod_copy<N>(ta).perform(true, 1.0, tb);

        // Compare against the reference

        compare_ref<N>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<size_t N>
void tod_copy_test::test_plain_additive(const dimensions<N> &dims, double d)
    throw (libtest::test_exception) {

    static const char *testname = "tod_copy_test::test_plain_additive()";

    try {

        dense_tensor<N, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);

        {
            dense_tensor_ctrl<N, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

            double *dta = tca.req_dataptr();
            double *dtb1 = tcb.req_dataptr();
            double *dtb2 = tcb_ref.req_dataptr();

            // Fill in random data

            abs_index<N> aida(dims);
            do {
                size_t i = aida.get_abs_index();
                dta[i] = drand48();
                dtb1[i] = drand48();
                dtb2[i] = dtb1[i] + d * dta[i];
            } while(aida.inc());
            tca.ret_dataptr(dta);
            dta = NULL;
            tcb.ret_dataptr(dtb1);
            dtb1 = NULL;
            tcb_ref.ret_dataptr(dtb2);
            dtb2 = NULL;
            ta.set_immutable();
            tb_ref.set_immutable();
        }

        // Invoke the copy operation

        tod_copy<N>(ta).perform(false, d, tb);

        // Compare against the reference

        compare_ref<N>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<size_t N>
void tod_copy_test::test_scaled(const dimensions<N> &dims, double c)
    throw (libtest::test_exception) {

    static const char *testname = "tod_copy_test::test_scaled()";

    try {

        dense_tensor<N, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);

        {
            dense_tensor_ctrl<N, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

            double *dta = tca.req_dataptr();
            double *dtb1 = tcb.req_dataptr();
            double *dtb2 = tcb_ref.req_dataptr();

            // Fill in random data

            abs_index<N> aida(dims);
            do {
                size_t i = aida.get_abs_index();
                dta[i] = dtb2[i] = drand48();
                dtb2[i] *= c;
                dtb1[i] = drand48();
            } while(aida.inc());
            tca.ret_dataptr(dta);
            dta = NULL;
            tcb.ret_dataptr(dtb1);
            dtb1 = NULL;
            tcb_ref.ret_dataptr(dtb2);
            dtb2 = NULL;
            ta.set_immutable();
            tb_ref.set_immutable();
        }

        // Invoke the copy operation

        tod_copy<N>(ta, c).perform(true, 1.0, tb);

        // Compare against the reference

        std::ostringstream ss;
        ss << "tod_copy_test::test_scaled(" << c << ")";
        compare_ref<N>::compare(ss.str().c_str(), tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<size_t N>
void tod_copy_test::test_scaled_additive(const dimensions<N> &dims, double c,
    double d) throw (libtest::test_exception) {

    static const char *testname = "tod_copy_test::test_scaled_additive()";

    try {

        dense_tensor<N, double, allocator_t> ta(dims), tb(dims), tb_ref(dims);

        {
            dense_tensor_ctrl<N, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

            double *dta = tca.req_dataptr();
            double *dtb1 = tcb.req_dataptr();
            double *dtb2 = tcb_ref.req_dataptr();

            // Fill in random data

            abs_index<N> aida(dims);
            do {
                size_t i = aida.get_abs_index();
                dta[i] = drand48();
                dtb1[i] = drand48();
                dtb2[i] = dtb1[i] + c * d * dta[i];
            } while(aida.inc());
            tca.ret_dataptr(dta);
            dta = NULL;
            tcb.ret_dataptr(dtb1);
            dtb1 = NULL;
            tcb_ref.ret_dataptr(dtb2);
            dtb2 = NULL;
            ta.set_immutable();
            tb_ref.set_immutable();
        }

        // Invoke the copy operation

        tod_copy<N>(ta, c).perform(false, d, tb);

        // Compare against the reference

        std::ostringstream ss;
        ss << "tod_copy_test::test_scaled_additive(" << c << ")";
        compare_ref<N>::compare(ss.str().c_str(), tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<size_t N>
void tod_copy_test::test_perm(const dimensions<N> &dims,
    const permutation<N> &perm) throw (libtest::test_exception) {

    static const char *testname = "tod_copy_test::test_perm()";

    try {

        dimensions<N> dimsa(dims), dimsb(dims);
        dimsb.permute(perm);

        dense_tensor<N, double, allocator_t> ta(dimsa), tb(dimsb),
            tb_ref(dimsb);

        {
            dense_tensor_ctrl<N, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

            double *dta = tca.req_dataptr();
            double *dtb1 = tcb.req_dataptr();
            double *dtb2 = tcb_ref.req_dataptr();

            // Fill in random data

            abs_index<N> aida(dimsa);
            do {
                index<N> ida(aida.get_index());
                index<N> idb(ida);
                idb.permute(perm);
                abs_index<N> aidb(idb, dimsb);
                size_t i, j;
                i = aida.get_abs_index();
                j = aidb.get_abs_index();
                dta[i] = dtb2[j] = drand48();
                dtb1[i] = drand48();
            } while(aida.inc());
            tca.ret_dataptr(dta);
            dta = NULL;
            tcb.ret_dataptr(dtb1);
            dtb1 = NULL;
            tcb_ref.ret_dataptr(dtb2);
            dtb2 = NULL;
            ta.set_immutable();
            tb_ref.set_immutable();
        }

        // Invoke the copy operation

        tod_copy<N>(ta, perm).perform(true, 1.0, tb);

        // Compare against the reference

        compare_ref<N>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<size_t N>
void tod_copy_test::test_perm_additive(const dimensions<N> &dims,
    const permutation<N> &perm, double d) throw (libtest::test_exception) {

    static const char *testname = "tod_copy_test::test_perm_additive()";

    try {

        dimensions<N> dimsa(dims), dimsb(dims);
        dimsb.permute(perm);

        dense_tensor<N, double, allocator_t> ta(dimsa), tb(dimsb),
            tb_ref(dimsb);

        {
            dense_tensor_ctrl<N, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

            double *dta = tca.req_dataptr();
            double *dtb1 = tcb.req_dataptr();
            double *dtb2 = tcb_ref.req_dataptr();

            // Fill in random data

            abs_index<N> aida(dimsa);
            do {
                index<N> ida(aida.get_index());
                index<N> idb(ida);
                idb.permute(perm);
                abs_index<N> aidb(idb, dimsb);
                size_t i, j;
                i = aida.get_abs_index();
                j = aidb.get_abs_index();
                dta[i] = drand48();
                dtb1[j] = drand48();
                dtb2[j] = dtb1[j] + d * dta[i];
            } while(aida.inc());
            tca.ret_dataptr(dta);
            dta = NULL;
            tcb.ret_dataptr(dtb1);
            dtb1 = NULL;
            tcb_ref.ret_dataptr(dtb2);
            dtb2 = NULL;
            ta.set_immutable();
            tb_ref.set_immutable();
        }

        // Invoke the copy operation

        tod_copy<N>(ta, perm).perform(false, d, tb);

        // Compare against the reference

        compare_ref<N>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<size_t N>
void tod_copy_test::test_perm_scaled(const dimensions<N> &dims,
    const permutation<N> &perm, double c) throw (libtest::test_exception) {

    static const char *testname = "tod_copy_test::test_perm_scaled()";

    try {

        dimensions<N> dimsa(dims), dimsb(dims);
        dimsb.permute(perm);

        dense_tensor<N, double, allocator_t> ta(dimsa), tb(dimsb),
            tb_ref(dimsb);

        {
            dense_tensor_ctrl<N, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

            double *dta = tca.req_dataptr();
            double *dtb1 = tcb.req_dataptr();
            double *dtb2 = tcb_ref.req_dataptr();

            // Fill in random data

            abs_index<N> aida(dimsa);
            do {
                index<N> idb(aida.get_index());
                idb.permute(perm);
                abs_index<N> aidb(idb, dimsb);
                size_t i, j;
                i = aida.get_abs_index();
                j = aidb.get_abs_index();
                dta[i] = drand48();
                dtb1[j] = drand48();
                dtb2[j] = c * dta[i];
            } while(aida.inc());
            tca.ret_dataptr(dta);
            dta = NULL;
            tcb.ret_dataptr(dtb1);
            dtb1 = NULL;
            tcb_ref.ret_dataptr(dtb2);
            dtb2 = NULL;
            ta.set_immutable();
            tb_ref.set_immutable();
        }

        // Invoke the copy operation

        tensor_transf<N, double> tr(perm, scalar_transf<double>(c));
        tod_copy<N> cp(ta, tr);
        cp.perform(true, 1.0, tb);

        // Compare against the reference

        compare_ref<N>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<size_t N>
void tod_copy_test::test_perm_scaled_additive(const dimensions<N> &dims,
    const permutation<N> &perm, double c, double d)
    throw (libtest::test_exception) {

    static const char *testname = "tod_copy_test::test_perm_scaled_additive()";

    try {

        dimensions<N> dimsa(dims), dimsb(dims);
        dimsb.permute(perm);

        dense_tensor<N, double, allocator_t> ta(dimsa), tb(dimsb),
            tb_ref(dimsb);

        {
            dense_tensor_ctrl<N, double> tca(ta), tcb(tb), tcb_ref(tb_ref);

            double *dta = tca.req_dataptr();
            double *dtb1 = tcb.req_dataptr();
            double *dtb2 = tcb_ref.req_dataptr();

            // Fill in random data

            abs_index<N> aida(dimsa);
            do {
                index<N> idb(aida.get_index());
                idb.permute(perm);
                abs_index<N> aidb(idb, dimsb);
                size_t i, j;
                i = aida.get_abs_index();
                j = aidb.get_abs_index();
                dta[i] = drand48();
                dtb1[j] = drand48();
                dtb2[j] = dtb1[j] + c * d * dta[i];
            } while(aida.inc());
            tca.ret_dataptr(dta);
            dta = NULL;
            tcb.ret_dataptr(dtb1);
            dtb1 = NULL;
            tcb_ref.ret_dataptr(dtb2);
            dtb2 = NULL;
            ta.set_immutable();
            tb_ref.set_immutable();
        }

        // Invoke the copy operation

        tensor_transf<N, double> tr(perm, scalar_transf<double>(c));
        tod_copy<N> cp(ta, tr);
        cp.perform(false, d, tb);

        // Compare against the reference

        compare_ref<N>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


void tod_copy_test::test_exc() throw (libtest::test_exception) {

    index<4> i1, i2, i3;
    i2[0] = 2;
    i2[1] = 2;
    i2[2] = 2;
    i2[3] = 2;
    i3[0] = 3;
    i3[1] = 3;
    i3[2] = 3;
    i3[3] = 3;
    index_range<4> ir1(i1, i2), ir2(i1, i3);
    dimensions<4> dim1(ir1), dim2(ir2);
    tensor4 t1(dim1), t2(dim2);

    bool ok = false;
    try {
        tod_copy<4>(t1).perform(true, 1.0, t2);
    } catch(exception &e) {
        ok = true;
    }

    if(!ok) {
        fail_test("tod_copy_test::test_exc()", __FILE__, __LINE__,
            "Expected an exception with heterogeneous arguments");
    }
}


} // namespace libtensor

