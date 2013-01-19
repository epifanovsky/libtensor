#include <sstream>
#include <libvmm/cuda_allocator.h>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_add.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_copy_d2h.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_copy_h2d.h>
#include "../compare_ref.h"
#include "cuda_tod_add_test.h"

namespace libtensor {


void cuda_tod_add_test::perform() throw(libtest::test_exception) {

    test_exc();

    test_add_to_self_pqrs(2, 3, 4, 5);
    test_add_two_pqrs_pqrs(2, 3, 4, 5);
    test_add_two_pqrs_qprs(2, 3, 4, 5);
    test_add_two_pqrs_prsq(3, 1, 1, 1);
    test_add_two_pqrs_prsq(2, 3, 4, 5);
    test_add_two_pqrs_qpsr(2, 3, 4, 5);
    test_add_two_ijkl_kjli(1, 2, 13, 2, 0.5, -1.0);
    test_add_mult(3, 2, 5, 4);
}


void cuda_tod_add_test::test_exc() throw(libtest::test_exception) {

    static const char *testname = "cuda_tod_add_test::test_exc()";

    typedef std_allocator<double> std_allocator;
    typedef libvmm::cuda_allocator<double> cuda_allocator;

    try {

    index<4> i1, i2;
    i2[0] = 2;
    i2[1] = 3;
    i2[2] = 5;
    i2[3] = 4;
    index_range<4> ir(i1, i2);
    dimensions<4> dim(ir);
    permutation<4> p1;
    p1.permute(0, 1);

    dense_tensor<4, double, std_allocator> h_t1(dim), h_t2(dim);
    dense_tensor<4, double, cuda_allocator> d_t1(dim), d_t2(dim);
    cuda_tod_add<4> add(d_t1, p1, 0.4);

    bool ok = false;
    try {
        add.add_op(d_t2, 1.0);
    } catch(exception &e) {
        ok = true;
    }

    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected an exception due to heterogeneous operands");
    }

    ok = false;
    try {
        add.prefetch();
        add.perform(true, 1.0, d_t2);
    } catch(exception& e) {
        ok = true;
    }

    if(!ok) {
        fail_test(testname, __FILE__, __LINE__,
            "Expected an exception due to heterogeneous result tensor");
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


void cuda_tod_add_test::test_add_to_self_pqrs(size_t p, size_t q, size_t r, size_t s)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "cuda_tod_add_test::test_add_to_self_pqrs(" << p << "," << q << ","
        << r << "," << s << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> std_allocator;
    typedef libvmm::cuda_allocator<double> cuda_allocator;

    try {

    index<4> i1, i2;
    i2[0] = p;
    i2[1] = q;
    i2[2] = r;
    i2[3] = s;
    index_range<4> ir(i1, i2);
    dimensions<4> dim(ir);
    dense_tensor<4, double, std_allocator> h_tc(dim), h_ta(dim), tc_ref(dim);
    dense_tensor<4, double, cuda_allocator> d_tc(dim), d_ta(dim);

    {
        dense_tensor_ctrl<4, double> ctrla(h_ta), ctrlc_ref(tc_ref);

        double *ptra = ctrla.req_dataptr();
        for(size_t i = 0; i < dim.get_size(); i++) ptra[i] = drand48();
        ctrla.ret_dataptr(ptra); ptra = 0;

        double *ptrc_ref = ctrlc_ref.req_dataptr();
        const double *cptra = ctrla.req_const_dataptr();
        for(size_t i = 0; i < dim.get_size(); i++) {
            ptrc_ref[i] = 2.0 * cptra[i] + 0.5 * cptra[i];
        }
        ctrla.ret_const_dataptr(cptra); cptra = 0;
        ctrlc_ref.ret_dataptr(ptrc_ref); ptrc_ref = 0;
    }

    cuda_tod_copy_h2d<4>(h_ta).perform(d_ta);

    cuda_tod_add<4> add(d_ta, 2.0);
    add.add_op(d_ta, 0.5);
    add.prefetch();
    add.perform(true, 1.0, d_tc);

    cuda_tod_copy_d2h<4>(d_tc).perform(h_tc);

    compare_ref<4>::compare(tn.c_str(), h_tc, tc_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void cuda_tod_add_test::test_add_two_pqrs_pqrs(size_t p, size_t q, size_t r,
    size_t s) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "cuda_tod_add_test::test_add_to_pqrs_pqrs(" << p << "," << q << ","
        << r << "," << s << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> std_allocator;
    typedef libvmm::cuda_allocator<double> cuda_allocator;

    try {

    index<4> i1, i2;
    i2[0] = p;
    i2[1] = q;
    i2[2] = r;
    i2[3] = s;
    index_range<4> ir(i1, i2);
    dimensions<4> dim(ir);
    dense_tensor<4, double, std_allocator> h_t1(dim), h_t2(dim), t1_ref(dim);
    dense_tensor<4, double, cuda_allocator> d_t1(dim), d_t2(dim);

    {
        dense_tensor_ctrl<4, double> ctrl1(h_t1), ctrl2(h_t2), ctrl1_ref(t1_ref);

        double *ptr1 = ctrl1.req_dataptr();
        double *ptr1_ref = ctrl1_ref.req_dataptr();
        double *ptr2 = ctrl2.req_dataptr();
        for(size_t i = 0; i < dim.get_size(); i++) {
            ptr1[i] = ptr1_ref[i] = drand48();
        }
        for(size_t i = 0; i < dim.get_size(); i++) ptr2[i] = drand48();
        ctrl1.ret_dataptr(ptr1); ptr1 = 0;
        ctrl2.ret_dataptr(ptr2); ptr2 = 0;

        const double* cptr2 = ctrl2.req_const_dataptr();
        for(size_t i = 0; i < dim.get_size(); i++) {
            ptr1_ref[i] += 2.0 * cptr2[i];
        }
        ctrl1_ref.ret_dataptr(ptr1_ref); ptr1_ref = 0;
        ctrl2.ret_const_dataptr(cptr2); cptr2 = 0;
    }
    cuda_tod_copy_h2d<4>(h_t1).perform(d_t1);
    cuda_tod_copy_h2d<4>(h_t2).perform(d_t2);

    cuda_tod_add<4> add(d_t2, 2.0);
    add.prefetch();
    add.perform(false, 1.0, d_t1);

    cuda_tod_copy_d2h<4>(d_t1).perform(h_t1);

    compare_ref<4>::compare(tn.c_str(), h_t1, t1_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void cuda_tod_add_test::test_add_two_pqrs_qprs(size_t p, size_t q, size_t r,
    size_t s) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "cuda_tod_add_test::test_add_two_pqrs_qprs(" << p << "," << q << ","
        << r << "," << s << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> std_allocator;
    typedef libvmm::cuda_allocator<double> cuda_allocator;

    try {

    index<4> i1, i2;
    i2[0] = p;
    i2[1] = q;
    i2[2] = r;
    i2[3] = s;
    index_range<4> ir(i1, i2);
    dimensions<4> dim1(ir), dim2(ir);
    permutation<4> p1, p2;
    p2.permute(0, 1);
    dim2.permute(p2);

    dense_tensor<4, double, std_allocator> h_t1(dim1), h_t2(dim2), t1_ref(dim1);
    dense_tensor<4, double, cuda_allocator> d_t1(dim1), d_t2(dim2);

    {
        dense_tensor_ctrl<4, double> ctrl1(h_t1), ctrl2(h_t2), ctrl1_ref(t1_ref);

        double *ptr1 = ctrl1.req_dataptr();
        double *ptr1_ref = ctrl1_ref.req_dataptr();
        double *ptr2 = ctrl2.req_dataptr();

        for(size_t i = 0; i < dim1.get_size(); i++) {
            ptr1[i] = ptr1_ref[i] = drand48();
        }
        for(size_t i = 0; i < dim2.get_size(); i++) ptr2[i] = drand48();

        ctrl1.ret_dataptr(ptr1); ptr1 = 0;
        ctrl2.ret_dataptr(ptr2); ptr2 = 0;

        const double* cptr2 = ctrl2.req_const_dataptr();
        size_t cnt = 0;
        for(size_t i = 0; i < dim1[0]; i++)
        for(size_t j = 0; j < dim1[1]; j++)
        for(size_t k = 0; k < dim1[2]; k++)
        for(size_t l = 0; l < dim1[3]; l++) {
            i1[0] = j; i1[1] = i; i1[2] = k; i1[3] = l;
            abs_index<4> ai(i1, dim2);
            ptr1_ref[cnt] += 0.1 * cptr2[ai.get_abs_index()];
            cnt++;
        }
        ctrl1_ref.ret_dataptr(ptr1_ref); ptr1_ref = 0;
        ctrl2.ret_const_dataptr(cptr2); cptr2 = 0;
    }
    cuda_tod_copy_h2d<4>(h_t1).perform(d_t1);
    cuda_tod_copy_h2d<4>(h_t2).perform(d_t2);

    cuda_tod_add<4> add(d_t2, p2, 0.1);
    add.prefetch();
    add.perform(false, 1.0, d_t1);

    cuda_tod_copy_d2h<4>(d_t1).perform(h_t1);

    compare_ref<4>::compare(tn.c_str(), h_t1, t1_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void cuda_tod_add_test::test_add_two_pqrs_prsq(size_t p, size_t q, size_t r,
    size_t s) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "cuda_tod_add_test::test_add_two_pqrs_prsq(";
    tnss << p << "," << q << "," << r << "," << s << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> std_allocator;
    typedef libvmm::cuda_allocator<double> cuda_allocator;

    try {

    index<4> i1, i2;
    i2[0] = p;
    i2[1] = q;
    i2[2] = r;
    i2[3] = s;
    index_range<4> ir(i1, i2);
    dimensions<4> dim1(ir), dim2(ir);
    permutation<4> p2;
    p2.permute(1, 2);
    p2.permute(2, 3);
    dim2.permute(p2);
    p2.invert();

    dense_tensor<4, double, std_allocator> h_t1(dim1), h_t2(dim2), t1_ref(dim1);
    dense_tensor<4, double, cuda_allocator> d_t1(dim1), d_t2(dim2);

    {
        dense_tensor_ctrl<4, double> ctrl1(h_t1), ctrl2(h_t2), ctrl1_ref(t1_ref);

        double *ptr1 = ctrl1.req_dataptr();
        double *ptr1_ref = ctrl1_ref.req_dataptr();
        double *ptr2 = ctrl2.req_dataptr();
        for(size_t i = 0; i < dim1.get_size(); i++) {
            ptr1[i] = ptr1_ref[i] = drand48();
        }
        for(size_t i = 0; i < dim1.get_size(); i++) ptr2[i] = drand48();
        ctrl1.ret_dataptr(ptr1); ptr1 = 0;
        ctrl2.ret_dataptr(ptr2); ptr2 = 0;

        const double* cptr2 = ctrl2.req_const_dataptr();
        size_t cnt = 0;
        for(size_t i = 0; i < dim1[0]; i++)
        for(size_t j = 0; j < dim1[1]; j++)
        for(size_t k = 0; k < dim1[2]; k++)
        for(size_t l = 0; l < dim1[3]; l++) {
            i1[0] = i; i1[1] = k; i1[2] = l; i1[3] = j;
            abs_index<4> ai(i1, dim2);
            ptr1_ref[cnt] += 0.1 * cptr2[ai.get_abs_index()];
            cnt++;
        }
        ctrl2.ret_const_dataptr(cptr2); cptr2 = 0;
        ctrl1_ref.ret_dataptr(ptr1_ref); ptr1_ref = 0;
    }

    cuda_tod_copy_h2d<4>(h_t1).perform(d_t1);
    cuda_tod_copy_h2d<4>(h_t2).perform(d_t2);

    cuda_tod_add<4> add(d_t2, p2, 0.1);
    add.prefetch();
    add.perform(false, 1.0, d_t1);

    cuda_tod_copy_d2h<4>(d_t1).perform(h_t1);

    compare_ref<4>::compare(tn.c_str(), h_t1, t1_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void cuda_tod_add_test::test_add_two_pqrs_qpsr(size_t p, size_t q, size_t r,
    size_t s) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "cuda_tod_add_test::test_add_two_pqrs_qpsr(";
    tnss << p << "," << q << "," << r << "," << s << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> std_allocator;
    typedef libvmm::cuda_allocator<double> cuda_allocator;

    try {

    index<4> i1, i2;
    i2[0] = p;
    i2[1] = q;
    i2[2] = r;
    i2[3] = s;
    index_range<4> ir(i1, i2);
    dimensions<4> dim1(ir), dim2(ir);
    permutation<4> p2;
    p2.permute(0, 1);
    p2.permute(2, 3);
    dim2.permute(p2);

    dense_tensor<4, double, std_allocator> h_t1(dim1), h_t2(dim2), t1_ref(dim1);
    dense_tensor<4, double, cuda_allocator> d_t1(dim1), d_t2(dim2);

    {
        dense_tensor_ctrl<4, double> ctrl1(h_t1), ctrl2(h_t2), ctrl1_ref(t1_ref);

        double *ptr1 = ctrl1.req_dataptr();
        double *ptr2 = ctrl2.req_dataptr();
        double *ptr1_ref = ctrl1_ref.req_dataptr();
        for(size_t i = 0; i < dim1.get_size(); i++) {
            ptr1[i] = ptr1_ref[i] = drand48();
        }
        for(size_t i = 0; i < dim2.get_size(); i++) {
            ptr2[i] = drand48();
        }
        ctrl1.ret_dataptr(ptr1); ptr1 = 0;
        ctrl2.ret_dataptr(ptr2); ptr2 = 0;

        const double *cptr2 = ctrl2.req_const_dataptr();
        size_t cnt = 0;
        for(size_t i = 0; i < dim1[0]; i++)
        for(size_t j = 0; j < dim1[1]; j++)
        for(size_t k = 0; k < dim1[2]; k++)
        for(size_t l = 0; l < dim1[3]; l++) {
            i1[0] = j; i1[1] = i; i1[2] = l; i1[3] = k;
            abs_index<4> ai(i1, dim2);
            ptr1_ref[cnt] += 0.1 * cptr2[ai.get_abs_index()];
            cnt++;
        }
        ctrl2.ret_const_dataptr(cptr2); cptr2 = 0;
        ctrl1_ref.ret_dataptr(ptr1_ref); ptr1_ref = 0;
    }

    cuda_tod_copy_h2d<4>(h_t1).perform(d_t1);
    cuda_tod_copy_h2d<4>(h_t2).perform(d_t2);

    cuda_tod_add<4> add(d_t2, p2, 0.1);
    add.prefetch();
    add.perform(false, 1.0, d_t1);

    cuda_tod_copy_d2h<4>(d_t1).perform(h_t1);

    compare_ref<4>::compare(tn.c_str(), h_t1, t1_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void cuda_tod_add_test::test_add_two_ijkl_kjli(size_t ni, size_t nj, size_t nk,
    size_t nl, double c1, double c2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "cuda_tod_add_test::test_add_two_ijkl_kjli(" << ni << ", " << nj << ", "
        << nk << ", " << nl << ", " << c1 << ", " << c2 << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> std_allocator;
    typedef libvmm::cuda_allocator<double> cuda_allocator;

    try {

    index<4> i1, i2;
    i2[0] = ni - 1; i2[1] = nj - 1; i2[2] = nk - 1; i2[3] = nl - 1;
    dimensions<4> dims_ijkl(index_range<4> (i1, i2));
    size_t sz = dims_ijkl.get_size();

    permutation<4> perm;
    perm.permute(0, 2).permute(2, 3); // ijkl->kjli

    dimensions<4> dims_kjli(dims_ijkl);
    dims_kjli.permute(perm);

    dense_tensor<4, double, std_allocator> h_t1(dims_ijkl), h_t2(dims_kjli),
        h_t3(dims_kjli), t3_ref(dims_kjli);
    dense_tensor<4, double, cuda_allocator> d_t1(dims_ijkl), d_t2(dims_kjli),
        d_t3(dims_kjli);

    dense_tensor_ctrl<4, double> ct1(h_t1), ct2(h_t2), ct3_ref(t3_ref);

    double *p1 = ct1.req_dataptr();
    double *p2 = ct2.req_dataptr();
    double *p3_ref = ct3_ref.req_dataptr();

    //    Generate random input

    for(size_t i = 0; i < sz; i++) {
        p1[i] = drand48();
        p2[i] = drand48();
    }

    //    Generate output reference data

    abs_index<4> ai(dims_ijkl);
    do {
        index<4> i1(ai.get_index()), i2(ai.get_index()), i3(ai.get_index());
        i2.permute(perm);
        i3.permute(perm);

        abs_index<4> ai1(i1, dims_ijkl), ai2(i2, dims_kjli), ai3(i3,
            dims_kjli);
        p3_ref[ai3.get_abs_index()] = c1 * p1[ai1.get_abs_index()] + c2
            * p2[ai2.get_abs_index()];
    } while(ai.inc());

    ct3_ref.ret_dataptr(p3_ref); p3_ref = 0;
    ct2.ret_dataptr(p2); p2 = 0;
    ct1.ret_dataptr(p1); p1 = 0;

    cuda_tod_copy_h2d<4>(h_t1).perform(d_t1);
    cuda_tod_copy_h2d<4>(h_t2).perform(d_t2);

    //  Invoke the operation

    cuda_tod_add<4> op(d_t1, perm, c1);
    op.add_op(d_t2, c2);
    op.perform(true, 1.0, d_t3);

    cuda_tod_copy_d2h<4>(d_t3).perform(h_t3);

    compare_ref<4>::compare(tn.c_str(), h_t3, t3_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void cuda_tod_add_test::test_add_mult(size_t p, size_t q, size_t r, size_t s)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "cuda_tod_add_test::test_add_mult(" << p << "," << q << "," << r << ","
        << s << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> std_allocator;
    typedef libvmm::cuda_allocator<double> cuda_allocator;

    try {

    index<4> i1, i2;
    i2[0] = p; i2[1] = q; i2[2] = r; i2[3] = s;
    index_range<4> ir(i1, i2);
    dimensions<4> dim(ir), dim3(ir);
    permutation<4> p3;
    p3.permute(0, 1);
    dim3.permute(p3);
    dense_tensor<4, double, std_allocator> h_t1(dim), h_t2(dim), h_t3(dim3), h_t4(dim),
        t1_ref(dim);
    dense_tensor<4, double, cuda_allocator> d_t1(dim), d_t2(dim), d_t3(dim3), d_t4(dim);

    {
        dense_tensor_ctrl<4, double> ctrl1(h_t1), ctrl2(h_t2), ctrl3(h_t3), ctrl4(h_t4),
            ctrl1_ref(t1_ref);

        double *ptr1 = ctrl1.req_dataptr();
        double *ptr2 = ctrl2.req_dataptr();
        double *ptr3 = ctrl3.req_dataptr();
        double *ptr4 = ctrl4.req_dataptr();
        double *ptr1_ref = ctrl1_ref.req_dataptr();
        for(size_t i = 0; i < dim.get_size(); i++) {
            ptr1[i] = ptr1_ref[i] = drand48();
        }
        for(size_t i = 0; i < dim.get_size(); i++) ptr2[i] = drand48();
        for(size_t i = 0; i < dim.get_size(); i++) ptr3[i] = drand48();
        for(size_t i = 0; i < dim.get_size(); i++) ptr4[i] = drand48();
        ctrl1.ret_dataptr(ptr1); ptr1 = 0;
        ctrl2.ret_dataptr(ptr2); ptr2 = 0;
        ctrl3.ret_dataptr(ptr3); ptr3 = 0;
        ctrl4.ret_dataptr(ptr4); ptr4 = 0;

        const double *cptr2 = ctrl2.req_const_dataptr();
        const double *cptr3 = ctrl3.req_const_dataptr();
        const double *cptr4 = ctrl4.req_const_dataptr();
        size_t cnt = 0;
        for(size_t i = 0; i < dim[0]; i++)
        for(size_t j = 0; j < dim[1]; j++)
        for(size_t k = 0; k < dim[2]; k++)
        for(size_t l = 0; l < dim[3]; l++) {
            i1[0] = j; i1[1] = i; i1[2] = k; i1[3] = l;
            abs_index<4> ai(i1, dim3);
            ptr1_ref[cnt] += 0.5 * (cptr2[cnt] -
                4.0 * cptr3[ai.get_abs_index()] + 0.2 * cptr4[cnt]);
            cnt++;
        }
        ctrl1_ref.ret_dataptr(ptr1_ref); ptr1_ref = 0;
        ctrl2.ret_const_dataptr(cptr2); cptr2 = 0;
        ctrl3.ret_const_dataptr(cptr3); cptr3 = 0;
        ctrl4.ret_const_dataptr(cptr4); cptr4 = 0;
    }

    cuda_tod_copy_h2d<4>(h_t1).perform(d_t1);
    cuda_tod_copy_h2d<4>(h_t2).perform(d_t2);
    cuda_tod_copy_h2d<4>(h_t3).perform(d_t3);
    cuda_tod_copy_h2d<4>(h_t4).perform(d_t4);

    cuda_tod_add<4> add(d_t2, 1.0);
    add.add_op(d_t3, p3, -4.0);
    add.add_op(d_t4, 0.2);
    add.prefetch();
    add.perform(false, 0.5, d_t1);

    cuda_tod_copy_d2h<4>(d_t1).perform(h_t1);

    compare_ref<4>::compare(tn.c_str(), h_t1, t1_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void cuda_tod_add_test::test_add_two_pq_qp(size_t p, size_t q)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "cuda_tod_add_test::test_add_two_pq_qp(" << p << "," << q << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> std_allocator;
    typedef libvmm::cuda_allocator<double> cuda_allocator;

    try {

    index<2> i1, i2;
    i2[0] = p;
    i2[1] = q;
    index_range<2> ir(i1, i2);
    dimensions<2> dim(ir), dim3(ir);
    permutation<2> perm, p3;
    p3.permute(0, 1);
    dim3.permute(p3);

    dense_tensor<2, double, std_allocator> h_t1(dim), h_t2(dim), h_t3(dim3), t1_ref(dim);
    dense_tensor<2, double, cuda_allocator> d_t1(dim), d_t2(dim), d_t3(dim3);

    {
        dense_tensor_ctrl<2, double> ctrl1(h_t1), ctrl2(h_t2), ctrl3(h_t3),
            ctrl1_ref(t1_ref);

        double *ptr1 = ctrl1.req_dataptr();
        double *ptr1_ref = ctrl1_ref.req_dataptr();
        double *ptr2 = ctrl2.req_dataptr();
        double *ptr3 = ctrl3.req_dataptr();
        for(size_t i = 0; i < dim.get_size(); i++) {
            ptr1[i] = ptr1_ref[i] = drand48();
        }
        for(size_t i = 0; i < dim.get_size(); i++) ptr2[i] = drand48();
        for(size_t i = 0; i < dim.get_size(); i++) ptr3[i] = drand48();
        ctrl1.ret_dataptr(ptr1); ptr1 = 0;
        ctrl2.ret_dataptr(ptr2); ptr2 = 0;
        ctrl3.ret_dataptr(ptr3); ptr3 = 0;

        const double *cptr2 = ctrl2.req_const_dataptr();
        const double *cptr3 = ctrl3.req_const_dataptr();
        size_t cnt = 0;
        for(size_t i = 0; i < dim[0]; i++)
        for(size_t j = 0; j < dim[1]; j++) {
            ptr1_ref[cnt] += 0.5 * (2.0 * cptr2[cnt] - cptr3[j * dim3[1] + i]);
            cnt++;
        }
        ctrl1_ref.ret_dataptr(ptr1_ref); ptr1_ref = 0;
        ctrl2.ret_const_dataptr(cptr2); cptr2 = 0;
        ctrl3.ret_const_dataptr(cptr3); cptr3 = 0;
    }

    cuda_tod_copy_h2d<2>(h_t1).perform(d_t1);
    cuda_tod_copy_h2d<2>(h_t2).perform(d_t2);
    cuda_tod_copy_h2d<2>(h_t3).perform(d_t3);

    cuda_tod_add<2> add(d_t2, 2.0);
    add.add_op(d_t3, p3, -1.0);
    add.prefetch();
    add.perform(false, 0.5, d_t1);

    cuda_tod_copy_d2h<2>(d_t1).perform(h_t1);

    compare_ref<2>::compare(tn.c_str(), h_t1, t1_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

