#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/print_dimensions.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_import_raw.h>
#include "../compare_ref.h"
#include "to_import_raw_test.h"

namespace libtensor {

void to_import_raw_test::perform() throw(libtest::test_exception) {
    std::cout << "Testing to_import_raw_test_x<double>   ";
    to_import_raw_test_x<double> t_double;
    t_double.perform();
    std::cout << "Testing to_import_raw_test_x<float>   ";
    to_import_raw_test_x<float> t_float;
    t_float.perform();
}

template<typename T>
void to_import_raw_test_x<T>::perform() throw(libtest::test_exception) {

    index<2> i2_s, i2_e, i2_m1, i2_m2;
    i2_e[0] = 9; i2_e[1] = 19;
    i2_m1[0] = 4; i2_m1[1] = 9;
    i2_m2[0] = 7; i2_m2[1] = 15;
    index_range<2> ir2_se(i2_s, i2_e), ir2_sm(i2_s, i2_m2),
        ir2_me(i2_m1, i2_e), ir2_mm(i2_m1, i2_m2);
    dimensions<2> dims2(ir2_se);

    index<4> i4_s, i4_e, i4_m1, i4_m2;
    i4_e[0] = 9; i4_e[1] = 19; i4_e[2] = 9; i4_e[3] = 19;
    i4_m1[0] = 4; i4_m1[1] = 9; i4_m1[2] = 4; i4_m1[3] = 9;
    i4_m2[0] = 7; i4_m2[1] = 15; i4_m2[2] = 7; i4_m2[3] = 15;
    index_range<4> ir4_se(i4_s, i4_e), ir4_sm(i4_s, i4_m2),
        ir4_me(i4_m1, i4_e), ir4_mm(i4_m1, i4_m2);
    dimensions<4> dims4(ir4_se);

    test_1(dims2, ir2_se);
    test_1(dims2, ir2_sm);
    test_1(dims2, ir2_me);
    test_1(dims2, ir2_mm);
    test_1(dims4, ir4_se);
    test_1(dims4, ir4_sm);
    test_1(dims4, ir4_me);
    test_1(dims4, ir4_mm);
}


template<typename T>
template<size_t N>
void to_import_raw_test_x<T>::test_1(const dimensions<N> &dims,
    const index_range<N> &ir) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_import_raw_test_x<T>::test_1(" << dims << ", "
        << ir.get_begin() << "->" << ir.get_end() << ")";

    typedef allocator<T> allocator_t;
    typedef dense_tensor<N, T, allocator_t> tensor_t;
    typedef dense_tensor_ctrl<N, T> tensor_ctrl_t;

    try {

    // Create tensors

    dimensions<N> dims_wnd(ir);
    size_t sz1 = dims.get_size(), sz2 = dims_wnd.get_size();

    tensor_t t1(dims), t2(dims_wnd), t2_ref(dims_wnd);

    {
    tensor_ctrl_t tc1(t1), tc2(t2), tc2_ref(t2_ref);

    // Fill in random data

    T *p1 = tc1.req_dataptr();
    for(size_t i = 0; i < sz1; i++) p1[i] = drand48();
    tc1.ret_dataptr(p1);
    t1.set_immutable();

    T *p2 = tc2.req_dataptr();
    for(size_t i = 0; i < sz2; i++) p2[i] = drand48();
    tc2.ret_dataptr(p2);

    // Create reference data

    const T *p1_ref = tc1.req_const_dataptr();
    T *p2_ref = tc2_ref.req_dataptr();
    abs_index<N> iwnd(dims_wnd);
    const index<N> &ibeg = ir.get_begin();
    do {
        index<N> idx;
        for(size_t i = 0; i < N; i++) {
        idx[i] = ibeg[i] + iwnd.get_index().at(i);
        }
        abs_index<N> aidx(idx, dims);
        p2_ref[iwnd.get_abs_index()] = p1_ref[aidx.get_abs_index()];
    } while(iwnd.inc());
    tc2_ref.ret_dataptr(p2_ref);
    tc1.ret_const_dataptr(p1_ref);

    // Invoke the operation

    p1_ref = tc1.req_const_dataptr();
    to_import_raw<N, T>(p1_ref, dims, ir).perform(t2);
    tc1.ret_const_dataptr(p1_ref);
    }

    // Compare against the reference

    compare_ref_x<N, T>::compare(tnss.str().c_str(), t2, t2_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
