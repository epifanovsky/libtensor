#include <algorithm>
#include <cmath>
#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/tod/tod_add.h>
#include "../compare_ref.h"
#include "tod_add_test.h"

namespace libtensor {


const double tod_add_test::k_thresh = 1e-14;


void tod_add_test::perform() throw(libtest::test_exception) {

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


void tod_add_test::test_exc() throw(libtest::test_exception) {

    static const char *testname = "tod_add_test::test_exc()";

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

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

    tensor<4, double, allocator> t1(dim), t2(dim);
    tod_add<4> add(t1, p1, 0.4);

    bool ok = false;
    try {
        add.add_op(t2, 1.0);
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
        add.perform(cpus, true, 1.0, t2);
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


void tod_add_test::test_add_to_self_pqrs(size_t p, size_t q, size_t r, size_t s)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_add_test::test_add_to_self_pqrs(" << p << "," << q << ","
        << r << "," << s << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

    try {

    index<4> i1, i2;
    i2[0] = p;
    i2[1] = q;
    i2[2] = r;
    i2[3] = s;
    index_range<4> ir(i1, i2);
    dimensions<4> dim(ir);
    tensor<4, double, allocator> tc(dim), ta(dim), tc_ref(dim);

    double ta_max = 0.0;
    {
        dense_tensor_ctrl<4, double> ctrla(ta), ctrlc_ref(tc_ref);

        double *ptra = ctrla.req_dataptr();
        for(size_t i = 0; i < dim.get_size(); i++) ptra[i] = drand48();
        ctrla.ret_dataptr(ptra); ptra = 0;

        double *ptrc_ref = ctrlc_ref.req_dataptr();
        const double *cptra = ctrla.req_const_dataptr();
        for(size_t i = 0; i < dim.get_size(); i++) {
            ptrc_ref[i] = 2.0 * cptra[i] + 0.5 * cptra[i];
            ta_max = std::max(ta_max, fabs(cptra[i]));
        }
        ctrla.ret_const_dataptr(cptra); cptra = 0;
        ctrlc_ref.ret_dataptr(ptrc_ref); ptrc_ref = 0;
    }

    tod_add<4> add(ta, 2.0);
    add.add_op(ta, 0.5);
    add.prefetch();
    add.perform(cpus, true, 1.0, tc);

    compare_ref<4>::compare(tn.c_str(), tc, tc_ref, ta_max * k_thresh);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_add_test::test_add_two_pqrs_pqrs(size_t p, size_t q, size_t r,
    size_t s) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_add_test::test_add_to_pqrs_pqrs(" << p << "," << q << ","
        << r << "," << s << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

    try {

    index<4> i1, i2;
    i2[0] = p;
    i2[1] = q;
    i2[2] = r;
    i2[3] = s;
    index_range<4> ir(i1, i2);
    dimensions<4> dim(ir);
    tensor<4, double, allocator> t1(dim), t2(dim), t1_ref(dim);

    double t2_max = 0.0;
    {
        dense_tensor_ctrl<4, double> ctrl1(t1), ctrl2(t2), ctrl1_ref(t1_ref);

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
            t2_max = std::max(t2_max, fabs(cptr2[i]));
        }
        ctrl1_ref.ret_dataptr(ptr1_ref); ptr1_ref = 0;
        ctrl2.ret_const_dataptr(cptr2); cptr2 = 0;
    }

    tod_add<4> add(t2, 2.0);
    add.prefetch();
    add.perform(cpus, false, 1.0, t1);

    compare_ref<4>::compare(tn.c_str(), t1, t1_ref, t2_max * k_thresh);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_add_test::test_add_two_pqrs_qprs(size_t p, size_t q, size_t r,
    size_t s) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_add_test::test_add_two_pqrs_qprs(" << p << "," << q << ","
        << r << "," << s << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

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

    tensor<4, double, allocator> t1(dim1), t2(dim2), t1_ref(dim1);

    double t2_max = 0.0;
    {
        dense_tensor_ctrl<4, double> ctrl1(t1), ctrl2(t2), ctrl1_ref(t1_ref);

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
            t2_max = std::max(t2_max, fabs(cptr2[ai.get_abs_index()]));
            cnt++;
        }
        ctrl1_ref.ret_dataptr(ptr1_ref); ptr1_ref = 0;
        ctrl2.ret_const_dataptr(cptr2); cptr2 = 0;
    }

    tod_add<4> add(t2, p2, 0.1);
    add.prefetch();
    add.perform(cpus, false, 1.0, t1);

    compare_ref<4>::compare(tn.c_str(), t1, t1_ref, t2_max * k_thresh);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_add_test::test_add_two_pqrs_prsq(size_t p, size_t q, size_t r,
    size_t s) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_add_test::test_add_two_pqrs_prsq(";
    tnss << p << "," << q << "," << r << "," << s << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

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

    tensor<4, double, allocator> t1(dim1), t2(dim2), t1_ref(dim1);

    double t2_max = 0.0;
    {
        dense_tensor_ctrl<4, double> ctrl1(t1), ctrl2(t2), ctrl1_ref(t1_ref);

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
            t2_max = std::max(t2_max, fabs(cptr2[ai.get_abs_index()]));
            cnt++;
        }
        ctrl2.ret_const_dataptr(cptr2); cptr2 = 0;
        ctrl1_ref.ret_dataptr(ptr1_ref); ptr1_ref = 0;
    }

    tod_add<4> add(t2, p2, 0.1);
    add.prefetch();
    add.perform(cpus, false, 1.0, t1);

    compare_ref<4>::compare(tn.c_str(), t1, t1_ref, t2_max * k_thresh);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_add_test::test_add_two_pqrs_qpsr(size_t p, size_t q, size_t r,
    size_t s) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_add_test::test_add_two_pqrs_qpsr(";
    tnss << p << "," << q << "," << r << "," << s << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

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

    tensor<4, double, allocator> t1(dim1), t2(dim2), t1_ref(dim1);

    double t2_max = 0.0;
    {
        dense_tensor_ctrl<4, double> ctrl1(t1), ctrl2(t2), ctrl1_ref(t1_ref);

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
            t2_max = std::max(t2_max, fabs(cptr2[ai.get_abs_index()]));
            cnt++;
        }
        ctrl2.ret_const_dataptr(cptr2); cptr2 = 0;
        ctrl1_ref.ret_dataptr(ptr1_ref); ptr1_ref = 0;
    }

    tod_add<4> add(t2, p2, 0.1);
    add.prefetch();
    add.perform(cpus, false, 1.0, t1);

    compare_ref<4>::compare(tn.c_str(), t1, t1_ref, t2_max * k_thresh);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_add_test::test_add_two_ijkl_kjli(size_t ni, size_t nj, size_t nk,
    size_t nl, double c1, double c2) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_add_test::test_add_two_ijkl_kjli(" << ni << ", " << nj << ", "
        << nk << ", " << nl << ", " << c1 << ", " << c2 << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

    try {

    index<4> i1, i2;
    i2[0] = ni - 1; i2[1] = nj - 1; i2[2] = nk - 1; i2[3] = nl - 1;
    dimensions<4> dims_ijkl(index_range<4> (i1, i2));
    size_t sz = dims_ijkl.get_size();

    permutation<4> perm;
    perm.permute(0, 2).permute(2, 3); // ijkl->kjli

    dimensions<4> dims_kjli(dims_ijkl);
    dims_kjli.permute(perm);

    tensor<4, double, allocator> t1(dims_ijkl), t2(dims_kjli),
        t3(dims_kjli), t3_ref(dims_kjli);

    dense_tensor_ctrl<4, double> ct1(t1), ct2(t2), ct3_ref(t3_ref);

    double *p1 = ct1.req_dataptr();
    double *p2 = ct2.req_dataptr();
    double *p3_ref = ct3_ref.req_dataptr();

    //	Generate random input

    for(size_t i = 0; i < sz; i++) {
        p1[i] = drand48();
        p2[i] = drand48();
    }

    //	Generate output reference data

    double t3_max = 0.0;
    abs_index<4> ai(dims_ijkl);
    do {
        index<4> i1(ai.get_index()), i2(ai.get_index()), i3(ai.get_index());
        i2.permute(perm);
        i3.permute(perm);

        abs_index<4> ai1(i1, dims_ijkl), ai2(i2, dims_kjli), ai3(i3,
            dims_kjli);
        p3_ref[ai3.get_abs_index()] = c1 * p1[ai1.get_abs_index()] + c2
            * p2[ai2.get_abs_index()];
        t3_max = std::max(t3_max, fabs(p3_ref[ai3.get_abs_index()]));
    } while(ai.inc());

    ct3_ref.ret_dataptr(p3_ref); p3_ref = 0;
    ct2.ret_dataptr(p2); p2 = 0;
    ct1.ret_dataptr(p1); p1 = 0;

    //	Invoke the operation

    tod_add<4> op(t1, perm, c1);
    op.add_op(t2, c2);
    op.perform(cpus, true, 1.0, t3);

    compare_ref<4>::compare(tn.c_str(), t3, t3_ref, t3_max * k_thresh);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_add_test::test_add_mult(size_t p, size_t q, size_t r, size_t s)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_add_test::test_add_mult(" << p << "," << q << "," << r << ","
        << s << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

    try {

    index<4> i1, i2;
    i2[0] = p; i2[1] = q; i2[2] = r; i2[3] = s;
    index_range<4> ir(i1, i2);
    dimensions<4> dim(ir), dim3(ir);
    permutation<4> p3;
    p3.permute(0, 1);
    dim3.permute(p3);
    tensor<4, double, allocator> t1(dim), t2(dim), t3(dim3), t4(dim),
        t1_ref(dim);

    double t_max = 0.0;
    {
        dense_tensor_ctrl<4, double> ctrl1(t1), ctrl2(t2), ctrl3(t3), ctrl4(t4),
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
            t_max = std::max(t_max, fabs(ptr1_ref[cnt]));
            cnt++;
        }
        ctrl1_ref.ret_dataptr(ptr1_ref); ptr1_ref = 0;
        ctrl2.ret_const_dataptr(cptr2); cptr2 = 0;
        ctrl3.ret_const_dataptr(cptr3); cptr3 = 0;
        ctrl4.ret_const_dataptr(cptr4); cptr4 = 0;
    }

    tod_add<4> add(t2, 1.0);
    add.add_op(t3, p3, -4.0);
    add.add_op(t4, 0.2);
    add.prefetch();
    add.perform(cpus, false, 0.5, t1);

    compare_ref<4>::compare(tn.c_str(), t1, t1_ref, t_max * k_thresh);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_add_test::test_add_two_pq_qp(size_t p, size_t q)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_add_test::test_add_two_pq_qp(" << p << "," << q << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator;

    cpu_pool cpus(1);

    try {

    index<2> i1, i2;
    i2[0] = p;
    i2[1] = q;
    index_range<2> ir(i1, i2);
    dimensions<2> dim(ir), dim3(ir);
    permutation<2> perm, p3;
    p3.permute(0, 1);
    dim3.permute(p3);
    tensor<2, double, allocator> t1(dim), t2(dim), t3(dim3), t1_ref(dim);

    double t_max = 0.0;
    {
        dense_tensor_ctrl<2, double> ctrl1(t1), ctrl2(t2), ctrl3(t3),
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
            t_max = std::max(t_max, fabs(ptr1_ref[cnt]));
            cnt++;
        }
        ctrl1_ref.ret_dataptr(ptr1_ref); ptr1_ref = 0;
        ctrl2.ret_const_dataptr(cptr2); cptr2 = 0;
        ctrl3.ret_const_dataptr(cptr3); cptr3 = 0;
    }

    tod_add<2> add(t2, 2.0);
    add.add_op(t3, p3, -1.0);
    add.prefetch();
    add.perform(cpus, false, 0.5, t1);

    compare_ref<2>::compare(tn.c_str(), t1, t1_ref, t_max * k_thresh);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
