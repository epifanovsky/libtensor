#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/to_apply.h>
#include <libtensor/dense_tensor/impl/to_apply_impl.h>
#include "../compare_ref.h"
#include "to_apply_test.h"

namespace libtensor {


void to_apply_test::perform() throw(libtest::test_exception) {
     std::cout << "Testing to_apply<double>  ";
     to_apply_test_x<double> t_double;
     t_double.perform();
     std::cout << "Testing to_apply<float>  ";
     to_apply_test_x<float> t_float;
     t_float.perform();
}



namespace to_apply_test_ns {

template<typename T>
struct sin_functor {
    T operator()(const T &x) { return std::sin(x); }
};

}

template<typename T>
void to_apply_test_x<T>::perform() throw(libtest::test_exception) {

    test_exc();

    to_apply_test_ns::sin_functor<T> sin;
    index<2> i2a, i2b; i2b[0]=10; i2b[1]=12;
    index_range<2> ir2(i2a, i2b); dimensions<2> dims2(ir2);
    permutation<2> perm2, perm2t;
    perm2t.permute(0, 1);

    test_plain(sin, dims2);
    test_plain_additive(sin, dims2, 1.0);
    test_plain_additive(sin, dims2, -1.0);
    test_plain_additive(sin, dims2, 2.5);

    test_scaled(sin, dims2, 1.0);
    test_scaled(sin, dims2, 0.5);
    test_scaled(sin, dims2, -3.14);
    test_scaled_additive(sin, dims2, 1.0, 1.0);
    test_scaled_additive(sin, dims2, 0.5, 1.0);
    test_scaled_additive(sin, dims2, -3.14, 1.0);
    test_scaled_additive(sin, dims2, 1.0, -1.0);
    test_scaled_additive(sin, dims2, 0.5, -1.0);
    test_scaled_additive(sin, dims2, -3.14, -1.0);
    test_scaled_additive(sin, dims2, 1.0, 2.5);
    test_scaled_additive(sin, dims2, 0.5, 2.5);
    test_scaled_additive(sin, dims2, -3.14, 2.5);

    test_perm(sin, dims2, perm2);
    test_perm(sin, dims2, perm2t);
    test_perm_additive(sin, dims2, perm2, 1.0);
    test_perm_additive(sin, dims2, perm2, -1.0);
    test_perm_additive(sin, dims2, perm2, 2.5);
    test_perm_additive(sin, dims2, perm2t, 1.0);
    test_perm_additive(sin, dims2, perm2t, -1.0);
    test_perm_additive(sin, dims2, perm2t, 2.5);

    test_perm_scaled(sin, dims2, perm2, 1.0);
    test_perm_scaled(sin, dims2, perm2t, 1.0);
    test_perm_scaled(sin, dims2, perm2, 0.5);
    test_perm_scaled(sin, dims2, perm2t, 0.5);
    test_perm_scaled(sin, dims2, perm2, -3.14);
    test_perm_scaled(sin, dims2, perm2t, -3.14);
    test_perm_scaled_additive(sin, dims2, perm2, 1.0, 1.0);
    test_perm_scaled_additive(sin, dims2, perm2t, 1.0, 1.0);
    test_perm_scaled_additive(sin, dims2, perm2, 0.5, 1.0);
    test_perm_scaled_additive(sin, dims2, perm2t, 0.5, 1.0);
    test_perm_scaled_additive(sin, dims2, perm2, -3.14, 1.0);
    test_perm_scaled_additive(sin, dims2, perm2t, -3.14, 1.0);
    test_perm_scaled_additive(sin, dims2, perm2, 1.0, -1.0);
    test_perm_scaled_additive(sin, dims2, perm2t, 1.0, -1.0);
    test_perm_scaled_additive(sin, dims2, perm2, 0.5, -1.0);
    test_perm_scaled_additive(sin, dims2, perm2t, 0.5, -1.0);
    test_perm_scaled_additive(sin, dims2, perm2, -3.14, -1.0);
    test_perm_scaled_additive(sin, dims2, perm2t, -3.14, -1.0);
    test_perm_scaled_additive(sin, dims2, perm2, 1.0, 2.5);
    test_perm_scaled_additive(sin, dims2, perm2t, 1.0, 2.5);
    test_perm_scaled_additive(sin, dims2, perm2, 0.5, 2.5);
    test_perm_scaled_additive(sin, dims2, perm2t, 0.5, 2.5);
    test_perm_scaled_additive(sin, dims2, perm2, -3.14, 2.5);
    test_perm_scaled_additive(sin, dims2, perm2t, -3.14, 2.5);

    index<4> i4a, i4b;
    i4b[0] = 4; i4b[1] = 5; i4b[2] = 6; i4b[3] = 7;
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    permutation<4> perm4, perm4c;
    perm4c.permute(0, 1).permute(1, 2).permute(2, 3);

    test_perm(sin, dims4, perm4);
    test_perm(sin, dims4, perm4c);

}

template<typename T>
template<size_t N, typename Functor>
void to_apply_test_x<T>::test_plain(Functor &fn, const dimensions<N> &dims)
    throw(libtest::test_exception) {

    static const char *testname = "to_apply_test_x<T>::test_plain()";

    typedef allocator<T> allocator;

    try {

    dense_tensor<N, T, allocator> ta(dims), tb(dims), tb_ref(dims);

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
        dtb2[i] = fn(dta[i]);
        dtb1[i] = drand48();
    } while(aida.inc());

    tca.ret_dataptr(dta); dta = NULL;
    tcb.ret_dataptr(dtb1); dtb1 = NULL;
    tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
    ta.set_immutable(); tb_ref.set_immutable();
    }

    // Invoke the operation

    to_apply<N, Functor, T> cp(ta, fn);
    cp.perform(true, tb);

    // Compare against the reference

    compare_ref_x<N, T>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}

template<typename T>
template<size_t N, typename Functor>
void to_apply_test_x<T>::test_plain_additive(Functor &fn,
    const dimensions<N> &dims, T d) throw(libtest::test_exception) {

    static const char *testname = "to_apply_test_x<T>::test_plain_additive()";

    typedef allocator<T> allocator;

    try {

    dense_tensor<N, T, allocator> ta(dims), tb(dims), tb_ref(dims);

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
        dtb2[i] = dtb1[i] + d * fn(dta[i]);
    } while(aida.inc());

    tca.ret_dataptr(dta); dta = NULL;
    tcb.ret_dataptr(dtb1); dtb1 = NULL;
    tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
    ta.set_immutable(); tb_ref.set_immutable();
    }

    // Invoke the operation

    permutation<N> p;
    tensor_transf<N, T> tr2(p, scalar_transf<T>(d));
    to_apply<N, Functor, T> cp(ta, fn, scalar_transf<T>(), tr2);
    cp.perform(false, tb);

    // Compare against the reference

    std::ostringstream ss;
    ss << "to_apply_test_x<T>::test_plain_additive(" << d << ")";
    compare_ref_x<N, T>::compare(ss.str().c_str(), tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}

template<typename T>
template<size_t N, typename Functor>
void to_apply_test_x<T>::test_scaled(Functor &fn,
    const dimensions<N> &dims, T c) throw(libtest::test_exception) {

    static const char *testname = "to_apply_test_x<T>::test_scaled()";

    typedef allocator<T> allocator;

    try {

    dense_tensor<N, T, allocator> ta(dims), tb(dims), tb_ref(dims);

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
        dtb2[i] = fn(c * dta[i]);
        dtb1[i] = drand48();
    } while(aida.inc());

    tca.ret_dataptr(dta); dta = NULL;
    tcb.ret_dataptr(dtb1); dtb1 = NULL;
    tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
    ta.set_immutable(); tb_ref.set_immutable();
    }

    // Invoke the operation

    to_apply<N, Functor, T> cp(ta, fn, c);
    cp.perform(true, tb);

    // Compare against the reference

    std::ostringstream ss; ss << "to_apply_test_x<T>::test_scaled(" << c << ")";
    compare_ref_x<N, T>::compare(ss.str().c_str(), tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}

template<typename T>
template<size_t N, typename Functor>
void to_apply_test_x<T>::test_scaled_additive(Functor &fn,
    const dimensions<N> &dims, T c, T d)
    throw(libtest::test_exception) {

    static const char *testname = "to_apply_test_x<T>::test_scaled_additive()";

    typedef allocator<T> allocator;

    try {

    dense_tensor<N, T, allocator> ta(dims), tb(dims), tb_ref(dims);

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
        dtb2[i] = dtb1[i] + d * fn(c * dta[i]);
    } while(aida.inc());

    tca.ret_dataptr(dta); dta = NULL;
    tcb.ret_dataptr(dtb1); dtb1 = NULL;
    tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
    ta.set_immutable(); tb_ref.set_immutable();
    }

    // Invoke the operation

    permutation<N> p;
    tensor_transf<N, T> tr2(p, scalar_transf<T>(d));
    to_apply<N, Functor, T> cp(ta, fn, scalar_transf<T>(c), tr2);
    cp.perform(false, tb);

    // Compare against the reference

    std::ostringstream ss; ss << "to_apply_test_x<T>::test_scaled_additive("
        << c << ")";
    compare_ref_x<N, T>::compare(ss.str().c_str(), tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}

template<typename T>
template<size_t N, typename Functor>
void to_apply_test_x<T>::test_perm(Functor &fn, const dimensions<N> &dims,
    const permutation<N> &perm) throw(libtest::test_exception) {

    static const char *testname = "to_apply_test_x<T>::test_perm()";

    typedef allocator<T> allocator;

    try {

    dimensions<N> dimsa(dims), dimsb(dims);
    dimsb.permute(perm);

    dense_tensor<N, T, allocator> ta(dimsa), tb(dimsb), tb_ref(dimsb);

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
        dtb1[i] = drand48();
        dtb2[j] = fn(dta[i]);
    } while(aida.inc());

    tca.ret_dataptr(dta); dta = NULL;
    tcb.ret_dataptr(dtb1); dtb1 = NULL;
    tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
    ta.set_immutable(); tb_ref.set_immutable();
    }

    // Invoke the operation

    to_apply<N, Functor, T> cp(ta, fn, perm);
    cp.perform(true, tb);

    // Compare against the reference

    compare_ref_x<N, T>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}

template<typename T>
template<size_t N, typename Functor>
void to_apply_test_x<T>::test_perm_additive(Functor &fn, const dimensions<N> &dims,
    const permutation<N> &perm, T d) throw(libtest::test_exception) {

    static const char *testname = "to_apply_test_x<T>::test_perm_additive()";

    typedef allocator<T> allocator;

    try {

    dimensions<N> dimsa(dims), dimsb(dims);
    dimsb.permute(perm);

    dense_tensor<N, T, allocator> ta(dimsa), tb(dimsb), tb_ref(dimsb);

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
        dtb2[j] = dtb1[j] + d * fn(dta[i]);
    } while(aida.inc());

    tca.ret_dataptr(dta); dta = NULL;
    tcb.ret_dataptr(dtb1); dtb1 = NULL;
    tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
    ta.set_immutable(); tb_ref.set_immutable();
    }

    // Invoke the operation

    tensor_transf<N, T> tr2(perm, scalar_transf<T>(d));
    to_apply<N, Functor, T> cp(ta, fn, scalar_transf<T>(), tr2);
    cp.perform(false, tb);

    // Compare against the reference

    compare_ref_x<N, T>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}

template<typename T>
template<size_t N, typename Functor>
void to_apply_test_x<T>::test_perm_scaled(Functor &fn, const dimensions<N> &dims,
    const permutation<N> &perm, T c) throw(libtest::test_exception) {

    static const char *testname = "to_apply_test_x<T>::test_perm_scaled()";

    typedef allocator<T> allocator;

    try {

    dimensions<N> dimsa(dims), dimsb(dims);
    dimsb.permute(perm);

    dense_tensor<N, T, allocator> ta(dimsa), tb(dimsb), tb_ref(dimsb);

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
        dtb2[j] = fn(c * dta[i]);
    } while(aida.inc());

    tca.ret_dataptr(dta); dta = NULL;
    tcb.ret_dataptr(dtb1); dtb1 = NULL;
    tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
    ta.set_immutable(); tb_ref.set_immutable();
    }

    // Invoke the operation

    to_apply<N, Functor, T> cp(ta, fn, perm, c);
    cp.perform(true, tb);

    // Compare against the reference

    compare_ref_x<N, T>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}

template<typename T>
template<size_t N, typename Functor>
void to_apply_test_x<T>::test_perm_scaled_additive(Functor &fn,
    const dimensions<N> &dims, const permutation<N> &perm, T c, T d)
    throw(libtest::test_exception) {

    static const char *testname =
        "to_apply_test_x<T>::test_perm_scaled_additive()";

    typedef allocator<T> allocator;

    try {

    dimensions<N> dimsa(dims), dimsb(dims);
    dimsb.permute(perm);

    dense_tensor<N, T, allocator> ta(dimsa), tb(dimsb), tb_ref(dimsb);

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
        dtb2[j] = dtb1[j] + d * fn(c * dta[i]);
    } while(aida.inc());

    tca.ret_dataptr(dta); dta = NULL;
    tcb.ret_dataptr(dtb1); dtb1 = NULL;
    tcb_ref.ret_dataptr(dtb2); dtb2 = NULL;
    ta.set_immutable(); tb_ref.set_immutable();
    }

    // Invoke the operation

    tensor_transf<N, T> tr2(perm, scalar_transf<T>(d));
    to_apply<N, Functor, T> cp(ta, fn, scalar_transf<T>(c), tr2);
    cp.perform(false, tb);

    // Compare against the reference

    compare_ref_x<N, T>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &exc) {
        fail_test(testname, __FILE__, __LINE__, exc.what());
    }
}

template<typename T>
void to_apply_test_x<T>::test_exc() throw(libtest::test_exception) {

    typedef allocator<T> allocator;

    index<4> i1, i2, i3;
    i2[0]=2; i2[1]=2; i2[2]=2; i2[3]=2;
    i3[0]=3; i3[1]=3; i3[2]=3; i3[3]=3;
    index_range<4> ir1(i1,i2), ir2(i1,i3);
    dimensions<4> dim1(ir1), dim2(ir2);
    dense_tensor<4, T, allocator> t1(dim1), t2(dim2);

    bool ok = false;
    try {
        typename to_apply_test_ns::sin_functor<T> sin;
        to_apply<4, to_apply_test_ns::sin_functor<T>, T> tc(t1, sin);
        tc.perform(true, t2);
    } catch(exception &e) {
        ok = true;
    }

    if(!ok) {
        fail_test("to_apply_test_x<T>::test_exc()", __FILE__, __LINE__,
        "Expected an exception with heterogeneous arguments");
    }
}


} // namespace libtensor

