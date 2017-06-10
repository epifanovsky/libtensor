#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_copy.h>
#include "../compare_ref.h"
#include "to_copy_test.h"

namespace libtensor {

//typedef allocator<T> allocator<T> ;
//typedef dense_tensor<4, T, allocator<T> > dense_tensor<4, T, allocator<T> > ;
//typedef dense_tensor_ctrl<4, T> dense_tensor_ctrl<4, T> ;


void to_copy_test::perform() throw (libtest::test_exception) {
     std::cout << "Testing to_copy<double>" << std::endl;
     perform_run<double>();
     std::cout << "Testing to_copy<float>" << std::endl;
     perform_run<float>();
}


template<typename T>
void to_copy_test::perform_run() throw (libtest::test_exception) {
    test_exc<T>();

    index<2> i2a, i2b;
    i2b[0] = 10;
    i2b[1] = 12;
    index_range<2> ir2(i2a, i2b);
    dimensions<2> dims2(ir2);
    permutation<2> perm2, perm2t;
    perm2t.permute(0, 1);

    test_plain<2, T>(dims2);
    test_plain_additive<2, T>(dims2, 1.0);
    test_plain_additive<2, T>(dims2, -1.0);
    test_plain_additive<2, T>(dims2, 2.5);

    test_scaled<2, T>(dims2, 1.0);
    test_scaled<2, T>(dims2, 0.5);
    test_scaled<2, T>(dims2, -3.14);
    test_scaled_additive<2, T>(dims2, 1.0, 1.0);
    test_scaled_additive<2, T>(dims2, 0.5, 1.0);
    test_scaled_additive<2, T>(dims2, -3.14, 1.0);
    test_scaled_additive<2, T>(dims2, 1.0, -1.0);
    test_scaled_additive<2, T>(dims2, 0.5, -1.0);
    test_scaled_additive<2, T>(dims2, -3.14, -1.0);
    test_scaled_additive<2, T>(dims2, 1.0, 2.5);
    test_scaled_additive<2, T>(dims2, 0.5, 2.5);
    test_scaled_additive<2, T>(dims2, -3.14, 2.5);

    test_perm<2, T>(dims2, perm2);
    test_perm<2, T>(dims2, perm2t);
    test_perm_additive<2, T>(dims2, perm2, 1.0);
    test_perm_additive<2, T>(dims2, perm2, -1.0);
    test_perm_additive<2, T>(dims2, perm2, 2.5);
    test_perm_additive<2, T>(dims2, perm2t, 1.0);
    test_perm_additive<2, T>(dims2, perm2t, -1.0);
    test_perm_additive<2, T>(dims2, perm2t, 2.5);

    test_perm_scaled<2, T>(dims2, perm2, 1.0);
    test_perm_scaled<2, T>(dims2, perm2t, 1.0);
    test_perm_scaled<2, T>(dims2, perm2, 0.5);
    test_perm_scaled<2, T>(dims2, perm2t, 0.5);
    test_perm_scaled<2, T>(dims2, perm2, -3.14);
    test_perm_scaled<2, T>(dims2, perm2t, -3.14);
    test_perm_scaled_additive<2, T>(dims2, perm2, 1.0, 1.0);
    test_perm_scaled_additive<2, T>(dims2, perm2t, 1.0, 1.0);
    test_perm_scaled_additive<2, T>(dims2, perm2, 0.5, 1.0);
    test_perm_scaled_additive<2, T>(dims2, perm2t, 0.5, 1.0);
    test_perm_scaled_additive<2, T>(dims2, perm2, -3.14, 1.0);
    test_perm_scaled_additive<2, T>(dims2, perm2t, -3.14, 1.0);
    test_perm_scaled_additive<2, T>(dims2, perm2, 1.0, -1.0);
    test_perm_scaled_additive<2, T>(dims2, perm2t, 1.0, -1.0);
    test_perm_scaled_additive<2, T>(dims2, perm2, 0.5, -1.0);
    test_perm_scaled_additive<2, T>(dims2, perm2t, 0.5, -1.0);
    test_perm_scaled_additive<2, T>(dims2, perm2, -3.14, -1.0);
    test_perm_scaled_additive<2, T>(dims2, perm2t, -3.14, -1.0);
    test_perm_scaled_additive<2, T>(dims2, perm2, 1.0, 2.5);
    test_perm_scaled_additive<2, T>(dims2, perm2t, 1.0, 2.5);
    test_perm_scaled_additive<2, T>(dims2, perm2, 0.5, 2.5);
    test_perm_scaled_additive<2, T>(dims2, perm2t, 0.5, 2.5);
    test_perm_scaled_additive<2, T>(dims2, perm2, -3.14, 2.5);
    test_perm_scaled_additive<2, T>(dims2, perm2t, -3.14, 2.5);

    index<4> i4a, i4b;
    i4b[0] = 4;
    i4b[1] = 5;
    i4b[2] = 6;
    i4b[3] = 7;
    dimensions<4> dims4(index_range<4> (i4a, i4b));
    permutation<4> perm4, perm4c;
    perm4c.permute(0, 1).permute(1, 2).permute(2, 3);

    test_perm<4, T>(dims4, perm4);
    test_perm<4, T>(dims4, perm4c);

}


template<size_t N, typename T>
void to_copy_test::test_plain(const dimensions<N> &dims)
    throw (libtest::test_exception) {

    static const char *testname = "to_copy_test::test_plain()";

    try {

        dense_tensor<N, T, allocator<T> > ta(dims), tb(dims), tb_ref(dims);

        {
            dense_tensor_ctrl<N, T> tca(ta), tcb(tb), tcb_ref(tb_ref);

            T *dta = tca.req_dataptr();
            T *dtb1 = tcb.req_dataptr();
            T *dtb2 = tcb_ref.req_dataptr();

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

        to_copy<N, T>(ta).perform(true, tb);

        // Compare against the reference

        compare_ref_x<N, T>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<size_t N, typename T>
void to_copy_test::test_plain_additive(const dimensions<N> &dims, T d)
    throw (libtest::test_exception) {

    static const char *testname = "to_copy_test::test_plain_additive()";

    try {

        dense_tensor<N, T, allocator<T> > ta(dims), tb(dims), tb_ref(dims);

        {
            dense_tensor_ctrl<N, T> tca(ta), tcb(tb), tcb_ref(tb_ref);

            T *dta = tca.req_dataptr();
            T *dtb1 = tcb.req_dataptr();
            T *dtb2 = tcb_ref.req_dataptr();

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

        to_copy<N, T>(ta, d).perform(false, tb);

        // Compare against the reference

        compare_ref_x<N, T>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<size_t N, typename T>
void to_copy_test::test_scaled(const dimensions<N> &dims, T c)
    throw (libtest::test_exception) {

    static const char *testname = "to_copy_test::test_scaled()";

    try {

        dense_tensor<N, T, allocator<T> > ta(dims), tb(dims), tb_ref(dims);

        {
            dense_tensor_ctrl<N, T> tca(ta), tcb(tb), tcb_ref(tb_ref);

            T *dta = tca.req_dataptr();
            T *dtb1 = tcb.req_dataptr();
            T *dtb2 = tcb_ref.req_dataptr();

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

        to_copy<N, T>(ta, c).perform(true, tb);

        // Compare against the reference

        std::ostringstream ss;
        ss << "to_copy_test::test_scaled(" << c << ")";
        compare_ref_x<N, T>::compare(ss.str().c_str(), tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<size_t N, typename T>
void to_copy_test::test_scaled_additive(const dimensions<N> &dims, T c,
    T d) throw (libtest::test_exception) {

    static const char *testname = "to_copy_test::test_scaled_additive()";

    try {

        dense_tensor<N, T, allocator<T> > ta(dims), tb(dims), tb_ref(dims);

        {
            dense_tensor_ctrl<N, T> tca(ta), tcb(tb), tcb_ref(tb_ref);

            T *dta = tca.req_dataptr();
            T *dtb1 = tcb.req_dataptr();
            T *dtb2 = tcb_ref.req_dataptr();

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

        to_copy<N, T>(ta, c * d).perform(false, tb);

        // Compare against the reference

        std::ostringstream ss;
        ss << "to_copy_test::test_scaled_additive(" << c << ")";
        compare_ref_x<N, T>::compare(ss.str().c_str(), tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<size_t N, typename T>
void to_copy_test::test_perm(const dimensions<N> &dims,
    const permutation<N> &perm) throw (libtest::test_exception) {

    static const char *testname = "to_copy_test::test_perm()";

    try {

        dimensions<N> dimsa(dims), dimsb(dims);
        dimsb.permute(perm);

        dense_tensor<N, T, allocator<T> > ta(dimsa), tb(dimsb),
            tb_ref(dimsb);

        {
            dense_tensor_ctrl<N, T> tca(ta), tcb(tb), tcb_ref(tb_ref);

            T *dta = tca.req_dataptr();
            T *dtb1 = tcb.req_dataptr();
            T *dtb2 = tcb_ref.req_dataptr();

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

        to_copy<N, T>(ta, perm).perform(true, tb);

        // Compare against the reference

        compare_ref_x<N, T>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<size_t N, typename T>
void to_copy_test::test_perm_additive(const dimensions<N> &dims,
    const permutation<N> &perm, T d) throw (libtest::test_exception) {

    static const char *testname = "to_copy_test::test_perm_additive()";

    try {

        dimensions<N> dimsa(dims), dimsb(dims);
        dimsb.permute(perm);

        dense_tensor<N, T, allocator<T> > ta(dimsa), tb(dimsb),
            tb_ref(dimsb);

        {
            dense_tensor_ctrl<N, T> tca(ta), tcb(tb), tcb_ref(tb_ref);

            T *dta = tca.req_dataptr();
            T *dtb1 = tcb.req_dataptr();
            T *dtb2 = tcb_ref.req_dataptr();

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

        to_copy<N, T>(ta, perm, d).perform(false, tb);

        // Compare against the reference

        compare_ref_x<N, T>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<size_t N, typename T>
void to_copy_test::test_perm_scaled(const dimensions<N> &dims,
    const permutation<N> &perm, T c) throw (libtest::test_exception) {

    static const char *testname = "to_copy_test::test_perm_scaled()";

    try {

        dimensions<N> dimsa(dims), dimsb(dims);
        dimsb.permute(perm);

        dense_tensor<N, T, allocator<T> > ta(dimsa), tb(dimsb),
            tb_ref(dimsb);

        {
            dense_tensor_ctrl<N, T> tca(ta), tcb(tb), tcb_ref(tb_ref);

            T *dta = tca.req_dataptr();
            T *dtb1 = tcb.req_dataptr();
            T *dtb2 = tcb_ref.req_dataptr();

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

        tensor_transf<N, T> tr(perm, scalar_transf<T>(c));
        to_copy<N, T> cp(ta, tr);
        cp.perform(true, tb);

        // Compare against the reference

        compare_ref_x<N, T>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<size_t N, typename T>
void to_copy_test::test_perm_scaled_additive(const dimensions<N> &dims,
    const permutation<N> &perm, T c, T d)
    throw (libtest::test_exception) {

    static const char *testname = "to_copy_test::test_perm_scaled_additive()";

    try {

        dimensions<N> dimsa(dims), dimsb(dims);
        dimsb.permute(perm);

        dense_tensor<N, T, allocator<T> > ta(dimsa), tb(dimsb),
            tb_ref(dimsb);

        {
            dense_tensor_ctrl<N, T> tca(ta), tcb(tb), tcb_ref(tb_ref);

            T *dta = tca.req_dataptr();
            T *dtb1 = tcb.req_dataptr();
            T *dtb2 = tcb_ref.req_dataptr();

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

        tensor_transf<N, T> tr(perm, scalar_transf<T>(c * d));
        to_copy<N, T> cp(ta, tr);
        cp.perform(false, tb);

        // Compare against the reference

        compare_ref_x<N, T>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}


template<typename T>
void to_copy_test::test_exc() throw (libtest::test_exception) {

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
    dense_tensor<4, T, allocator<T> >  t1(dim1), t2(dim2);

    bool ok = false;
    try {
        to_copy<4, T>(t1).perform(true, t2);
    } catch(exception &e) {
        ok = true;
    }

    if(!ok) {
        fail_test("to_copy_test::test_exc()", __FILE__, __LINE__,
            "Expected an exception with heterogeneous arguments");
    }
}



//template<typename double> void to_copy_test::perform_run();

} // namespace libtensor

