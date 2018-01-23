#include <cmath>
#include <ctime>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_extract.h>
#include "../compare_ref.h"
#include "to_extract_test.h"

namespace libtensor {

void to_extract_test::perform() throw(libtest::test_exception) {
    std::cout << "Testing to_extract_test_x<double>   ";
    to_extract_test_x<double> t_double;
    t_double.perform();
    std::cout << "Testing to_extract_test_x<float>   ";
    to_extract_test_x<float> t_float;
    t_float.perform();
}

template<typename T>
void to_extract_test_x<T>::perform() throw(libtest::test_exception) {

    srand48(time(0));

    test_1();
    test_2();
    test_3();
    test_4();
    test_5();
}


/** \test Extract a single matrix row: \f$ b_i = a_{ij} |_{j=2} \f$
 **/
template<typename T>
void to_extract_test_x<T>::test_1() throw(libtest::test_exception) {

    static const char *testname = "to_extract_test_x<T>::test_1()";

    typedef allocator<T> allocator;

    try {

    index<1> i1a, i1b;
    i1b[0] = 10;
    index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 10;
    dimensions<1> dims1(index_range<1>(i1a, i1b));
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    size_t sza = dims2.get_size(), szb = dims1.get_size();

    dense_tensor<2, T, allocator> ta(dims2);
    dense_tensor<1, T, allocator> tb(dims1), tb_ref(dims1);

    {
    dense_tensor_ctrl<2, T> tca(ta);
    dense_tensor_ctrl<1, T> tcb(tb), tcb_ref(tb_ref);

    T *pa = tca.req_dataptr();
    T *pb = tcb.req_dataptr();
    T *pb_ref = tcb_ref.req_dataptr();

    for(size_t i = 0; i < sza; i++) pa[i] = drand48();
    for(size_t i = 0; i < szb; i++) pb[i] = drand48();

    for(size_t i = 0; i < szb; i++) {
        index<2> idxa; idxa[0] = i; idxa[1] = 2;
        index<1> idxb; idxb[0] = i;
        abs_index<2> aidxa(idxa, dims2);
        abs_index<1> aidxb(idxb, dims1);
        pb_ref[aidxb.get_abs_index()] = pa[aidxa.get_abs_index()];
    }

    tca.ret_dataptr(pa); pa = 0;
    tcb.ret_dataptr(pb); pb = 0;
    tcb_ref.ret_dataptr(pb_ref); pb_ref = 0;
    }

    mask<2> m; m[0] = true; m[1] = false;
    index<2> idx; idx[0] = 0; idx[1] = 2;
    to_extract<2, 1, T>(ta, m, idx).perform(true, tb);

    compare_ref_x<1, T>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract a tensor slice with one index fixed:
        \f$ b_{ij} = a_{ikj} |_{k=0} \f$
 **/
template<typename T>
void to_extract_test_x<T>::test_2() throw(libtest::test_exception) {

    static const char *testname = "to_extract_test_x<T>::test_2()";

    typedef allocator<T> allocator;

    try {

    size_t ni = 6, nj = 11, nk = 3;
    index<2> i2a, i2b;
    i2b[0] = ni - 1; i2b[1] = nj - 1;
    index<3> i3a, i3b;
    i3b[0] = ni - 1; i3b[1] = nk - 1; i3b[2] = nj - 1;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<3> dims3(index_range<3>(i3a, i3b));
    size_t sza = dims3.get_size(), szb = dims2.get_size();

    dense_tensor<3, T, allocator> ta(dims3);
    dense_tensor<2, T, allocator> tb(dims2), tb_ref(dims2);

    {
    dense_tensor_ctrl<3, T> tca(ta);
    dense_tensor_ctrl<2, T> tcb(tb), tcb_ref(tb_ref);

    T *pa = tca.req_dataptr();
    T *pb = tcb.req_dataptr();
    T *pb_ref = tcb_ref.req_dataptr();

    for(size_t i = 0; i < sza; i++) pa[i] = drand48();
    for(size_t i = 0; i < szb; i++) pb[i] = drand48();

    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
        index<3> idxa; idxa[0] = i; idxa[1] = 0; idxa[2] = j;
        index<2> idxb; idxb[0] = i; idxb[1] = j;
        abs_index<3> aidxa(idxa, dims3);
        abs_index<2> aidxb(idxb, dims2);
        pb_ref[aidxb.get_abs_index()] = pa[aidxa.get_abs_index()];
    }
    }

    tca.ret_dataptr(pa); pa = 0;
    tcb.ret_dataptr(pb); pb = 0;
    tcb_ref.ret_dataptr(pb_ref); pb_ref = 0;
    }

    mask<3> m; m[0] = true; m[1] = false; m[2] = true;
    index<3> idx; idx[0] = 0; idx[1] = 0; idx[2] = 0;
    to_extract<3, 1, T>(ta, m, idx).perform(true, tb);

    compare_ref_x<2, T>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Extract a tensor slice with one index fixed and permutation:
        \f$ b_{ji} = a_{ikj} |_{k=0} \f$
 **/
template<typename T>
void to_extract_test_x<T>::test_3() throw(libtest::test_exception) {

    static const char *testname = "to_extract_test_x<T>::test_3()";

    typedef allocator<T> allocator;

    try {

    size_t ni = 6, nj = 11, nk = 3;
    index<2> i2a, i2b;
    i2b[0] = nj - 1; i2b[1] = ni - 1;
    index<3> i3a, i3b;
    i3b[0] = ni - 1; i3b[1] = nk - 1; i3b[2] = nj - 1;
    dimensions<2> dims2(index_range<2>(i2a, i2b));
    dimensions<3> dims3(index_range<3>(i3a, i3b));
    size_t sza = dims3.get_size(), szb = dims2.get_size();

    dense_tensor<3, T, allocator> ta(dims3);
    dense_tensor<2, T, allocator> tb(dims2), tb_ref(dims2);

    {
    dense_tensor_ctrl<3, T> tca(ta);
    dense_tensor_ctrl<2, T> tcb(tb), tcb_ref(tb_ref);

    T *pa = tca.req_dataptr();
    T *pb = tcb.req_dataptr();
    T *pb_ref = tcb_ref.req_dataptr();

    for(size_t i = 0; i < sza; i++) pa[i] = drand48();
    for(size_t i = 0; i < szb; i++) pb[i] = drand48();

    for(size_t i = 0; i < ni; i++) {
    for(size_t j = 0; j < nj; j++) {
        index<3> idxa; idxa[0] = i; idxa[1] = 0; idxa[2] = j;
        index<2> idxb; idxb[0] = j; idxb[1] = i;
        abs_index<3> aidxa(idxa, dims3);
        abs_index<2> aidxb(idxb, dims2);
        pb_ref[aidxb.get_abs_index()] = pa[aidxa.get_abs_index()];
    }
    }

    tca.ret_dataptr(pa); pa = 0;
    tcb.ret_dataptr(pb); pb = 0;
    tcb_ref.ret_dataptr(pb_ref); pb_ref = 0;
    }

    permutation<2> perm;
    perm.permute(0, 1);

    mask<3> m; m[0] = true; m[1] = false; m[2] = true;
    index<3> idx; idx[0] = 0; idx[1] = 0; idx[2] = 0;
    to_extract<3, 1, T>(ta, m,idx, perm).perform(true, tb);

    compare_ref_x<2, T>::compare(testname, tb, tb_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void to_extract_test_x<T>::test_4() throw(libtest::test_exception) {

    static const char *testname = "to_extract_test_x<T>::test_4()";

    typedef allocator<T> allocator;

    try {


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void to_extract_test_x<T>::test_5() throw(libtest::test_exception) {

    static const char *testname = "to_extract_test_x<T>::test_5()";

    typedef allocator<T> allocator;

    try {


    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
