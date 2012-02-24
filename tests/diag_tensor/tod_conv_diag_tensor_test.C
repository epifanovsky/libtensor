#include <cstdlib>
#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/diag_tensor/diag_tensor.h>
#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include <libtensor/diag_tensor/tod_conv_diag_tensor.h>
#include "../compare_ref.h"
#include "tod_conv_diag_tensor_test.h"

namespace libtensor {


void tod_conv_diag_tensor_test::perform() throw(libtest::test_exception) {

    test_1_i(1);
    test_1_i(4);
    test_1_i(21);
    test_2_i(1);
    test_2_i(4);
    test_2_i(21);
    test_3_i(1);
    test_3_i(4);
    test_3_i(21);

    test_1_ij(1, 1);
    test_1_ij(1, 4);
    test_1_ij(4, 1);
    test_1_ij(4, 4);
    test_1_ij(10, 21);
    test_1_ij(21, 10);
    test_2_ij(1, 1);
    test_2_ij(1, 4);
    test_2_ij(4, 1);
    test_2_ij(4, 4);
    test_2_ij(10, 21);
    test_2_ij(21, 10);
    test_1_ii(1);
    test_1_ii(4);
    test_1_ii(21);
    test_1_ii_ij(1);
    test_1_ii_ij(4);
    test_1_ii_ij(21);

    test_1_iik_iji(1);
    test_1_iik_iji(4);
    test_1_iik_iji(21);

    test_1_iijj_ijij_ijjk(1);
    test_1_iijj_ijij_ijjk(4);
    test_1_iijj_ijij_ijjk(21);
}


void tod_conv_diag_tensor_test::test_1_i(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_conv_diag_tensor_test::test_1_i(" << ni << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<1> i1, i2;
        i2[0] = ni - 1;
        dimensions<1> dims(index_range<1>(i1, i2));
        size_t sz = dims.get_size();

        diag_tensor_space<1> dts(dims);
        diag_tensor_subspace<1> dtss1(0);
        size_t ssn1 = dts.add_subspace(dtss1);

        diag_tensor<1, double, allocator_t> dt(dts);
        dense_tensor<1, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<1, double> cdt(dt);
            dense_tensor_wr_ctrl<1, double> ct_ref(t_ref);
            double *p1 = cdt.req_dataptr(ssn1);
            double *p2 = ct_ref.req_dataptr();
            for(size_t i = 0; i < sz; i++) p1[i] = p2[i] = drand48();
            cdt.ret_dataptr(ssn1, p1); p1 = 0;
            ct_ref.ret_dataptr(p2); p2 = 0;
        }

        tod_conv_diag_tensor<1>(dt).perform(t);
	compare_ref<1>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_conv_diag_tensor_test::test_2_i(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_conv_diag_tensor_test::test_2_i(" << ni << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<1> i1, i2;
        i2[0] = ni - 1;
        dimensions<1> dims(index_range<1>(i1, i2));
        size_t sz = dims.get_size();

        mask<1> m1;
        m1[0] = true;

        diag_tensor_space<1> dts(dims);
        diag_tensor_subspace<1> dtss1(1);
        dtss1.set_diag_mask(0, m1);
        size_t ssn1 = dts.add_subspace(dtss1);

        diag_tensor<1, double, allocator_t> dt(dts);
        dense_tensor<1, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<1, double> cdt(dt);
            dense_tensor_wr_ctrl<1, double> ct_ref(t_ref);
            double *p1 = cdt.req_dataptr(ssn1);
            double *p2 = ct_ref.req_dataptr();
            for(size_t i = 0; i < sz; i++) p1[i] = p2[i] = drand48();
            cdt.ret_dataptr(ssn1, p1); p1 = 0;
            ct_ref.ret_dataptr(p2); p2 = 0;
        }

        tod_conv_diag_tensor<1>(dt).perform(t);
	compare_ref<1>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_conv_diag_tensor_test::test_3_i(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_conv_diag_tensor_test::test_3_i(" << ni << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<1> i1, i2;
        i2[0] = ni - 1;
        dimensions<1> dims(index_range<1>(i1, i2));
        size_t sz = dims.get_size();

        mask<1> m1;
        m1[0] = true;

        diag_tensor_space<1> dts(dims);
        diag_tensor_subspace<1> dtss1(0), dtss2(1);
        dtss2.set_diag_mask(0, m1);
        size_t ssn1 = dts.add_subspace(dtss1);
        size_t ssn2 = dts.add_subspace(dtss2);

        diag_tensor<1, double, allocator_t> dt(dts);
        dense_tensor<1, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<1, double> cdt(dt);
            dense_tensor_wr_ctrl<1, double> ct(t), ct_ref(t_ref);
            double *p1 = cdt.req_dataptr(ssn1);
            double *p2 = cdt.req_dataptr(ssn2);
            double *p = ct.req_dataptr();
            double *p_ref = ct_ref.req_dataptr();
            for(size_t i = 0; i < sz; i++) {
                p1[i] = drand48();
                p2[i] = drand48();
                p[i] = drand48();
                p_ref[i] = p1[i] + p2[i];
            }
            cdt.ret_dataptr(ssn1, p1); p1 = 0;
            cdt.ret_dataptr(ssn2, p2); p2 = 0;
            ct_ref.ret_dataptr(p_ref); p_ref = 0;
            ct.ret_dataptr(p); p = 0;
        }

        tod_conv_diag_tensor<1>(dt).perform(t);
	compare_ref<1>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_conv_diag_tensor_test::test_1_ij(size_t ni, size_t nj)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_conv_diag_tensor_test::test_1_ij(" << ni << ", " << nj << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i1, i2;
        i2[0] = ni - 1; i2[1] = nj - 1;
        dimensions<2> dims(index_range<2>(i1, i2));
        size_t sz = dims.get_size();

        diag_tensor_space<2> dts(dims);
        diag_tensor_subspace<2> dtss1(0);
        size_t ssn1 = dts.add_subspace(dtss1);

        diag_tensor<2, double, allocator_t> dt(dts);
        dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<2, double> cdt(dt);
            dense_tensor_wr_ctrl<2, double> ct_ref(t_ref);
            double *p1 = cdt.req_dataptr(ssn1);
            double *p2 = ct_ref.req_dataptr();
            for(size_t i = 0; i < sz; i++) p1[i] = p2[i] = drand48();
            cdt.ret_dataptr(ssn1, p1); p1 = 0;
            ct_ref.ret_dataptr(p2); p2 = 0;
        }

        tod_conv_diag_tensor<2>(dt).perform(t);
	compare_ref<2>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_conv_diag_tensor_test::test_2_ij(size_t ni, size_t nj)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_conv_diag_tensor_test::test_2_ij(" << ni << ", " << nj << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i1, i2;
        i2[0] = ni - 1; i2[1] = nj - 1;
        dimensions<2> dims(index_range<2>(i1, i2));
        size_t sz = dims.get_size();

        mask<2> m01;
        m01[1] = true;

        diag_tensor_space<2> dts(dims);
        diag_tensor_subspace<2> dtss1(1);
        dtss1.set_diag_mask(0, m01);
        size_t ssn1 = dts.add_subspace(dtss1);

        diag_tensor<2, double, allocator_t> dt(dts);
        dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<2, double> cdt(dt);
            dense_tensor_wr_ctrl<2, double> ct_ref(t_ref);
            double *p1 = cdt.req_dataptr(ssn1);
            double *p2 = ct_ref.req_dataptr();
            for(size_t i = 0; i < sz; i++) p1[i] = p2[i] = drand48();
            cdt.ret_dataptr(ssn1, p1); p1 = 0;
            ct_ref.ret_dataptr(p2); p2 = 0;
        }

        tod_conv_diag_tensor<2>(dt).perform(t);
	compare_ref<2>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_conv_diag_tensor_test::test_1_ii(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_conv_diag_tensor_test::test_1_ii(" << ni << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i1, i2;
        i2[0] = ni - 1; i2[1] = ni - 1;
        dimensions<2> dims(index_range<2>(i1, i2));
        size_t sz1 = ni, sz2 = ni * ni;

        mask<2> m11;
        m11[0] = true; m11[1] = true;

        diag_tensor_space<2> dts(dims);
        diag_tensor_subspace<2> dtss1(1);
        dtss1.set_diag_mask(0, m11);
        size_t ssn1 = dts.add_subspace(dtss1);

        diag_tensor<2, double, allocator_t> dt(dts);
        dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<2, double> cdt(dt);
            dense_tensor_wr_ctrl<2, double> ct(t), ct_ref(t_ref);
            double *p1 = cdt.req_dataptr(ssn1);
            double *p = ct.req_dataptr();
            double *p_ref = ct_ref.req_dataptr();
            for(size_t i = 0; i < ni; i++) {
                size_t ii = i * ni + i;
                for(size_t j = 0; j < ni; j++) {
                    size_t ij = i * ni + j;
                    p_ref[ij] = 0.0;
                    p[ij] = drand48();
                }
                p1[i] = drand48();
                p_ref[ii] = p1[i];
            }
            cdt.ret_dataptr(ssn1, p1); p1 = 0;
            ct_ref.ret_dataptr(p_ref); p_ref = 0;
            ct.ret_dataptr(p); p = 0;
        }

        tod_conv_diag_tensor<2>(dt).perform(t);
	compare_ref<2>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_conv_diag_tensor_test::test_1_ii_ij(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_conv_diag_tensor_test::test_1_ii_ij(" << ni << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> i1, i2;
        i2[0] = ni - 1; i2[1] = ni - 1;
        dimensions<2> dims(index_range<2>(i1, i2));
        size_t sz1 = ni, sz2 = ni * ni;

        mask<2> m11;
        m11[0] = true; m11[1] = true;

        diag_tensor_space<2> dts(dims);
        diag_tensor_subspace<2> dtss1(1), dtss2(0);
        dtss1.set_diag_mask(0, m11);
        size_t ssn1 = dts.add_subspace(dtss1);
        size_t ssn2 = dts.add_subspace(dtss2);

        diag_tensor<2, double, allocator_t> dt(dts);
        dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<2, double> cdt(dt);
            dense_tensor_wr_ctrl<2, double> ct(t), ct_ref(t_ref);
            double *p1 = cdt.req_dataptr(ssn1);
            double *p2 = cdt.req_dataptr(ssn2);
            double *p = ct.req_dataptr();
            double *p_ref = ct_ref.req_dataptr();
            for(size_t i = 0; i < ni; i++) {
                size_t ii = i * ni + i;
                for(size_t j = 0; j < ni; j++) {
                    size_t ij = i * ni + j;
                    p2[ij] = p_ref[ij] = drand48();
                    p[ij] = drand48();
                }
                p1[i] = drand48();
                p_ref[ii] += p1[i];
            }
            cdt.ret_dataptr(ssn1, p1); p1 = 0;
            cdt.ret_dataptr(ssn2, p2); p2 = 0;
            ct_ref.ret_dataptr(p_ref); p_ref = 0;
            ct.ret_dataptr(p); p = 0;
        }

        tod_conv_diag_tensor<2>(dt).perform(t);
	compare_ref<2>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_conv_diag_tensor_test::test_1_iik_iji(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_conv_diag_tensor_test::test_1_iik_iji(" << ni << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<3> i1, i2;
        i2[0] = ni - 1; i2[1] = ni - 1; i2[2] = ni - 1;
        dimensions<3> dims(index_range<3>(i1, i2));
        size_t sz = dims.get_size();

        mask<3> m110, m101;
        m110[0] = true; m110[1] = true;
        m101[0] = true; m101[2] = true;

        diag_tensor_space<3> dts(dims);
        diag_tensor_subspace<3> dtss1(1), dtss2(1);
        dtss1.set_diag_mask(0, m110);
        dtss2.set_diag_mask(0, m101);
        size_t ssn1 = dts.add_subspace(dtss1);
        size_t ssn2 = dts.add_subspace(dtss2);

        diag_tensor<3, double, allocator_t> dt(dts);
        dense_tensor<3, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<3, double> cdt(dt);
            dense_tensor_wr_ctrl<3, double> ct(t), ct_ref(t_ref);

            double *p = ct.req_dataptr();
            double *p_ref = ct_ref.req_dataptr();
            for(size_t i = 0; i < sz; i++) {
                p[i] = drand48();
                p_ref[i] = 0.0;
            }
            ct.ret_dataptr(p); p = 0;

            double *p1 = cdt.req_dataptr(ssn1);
            for(size_t i = 0; i < ni; i++)
            for(size_t k = 0; k < ni; k++) {
                size_t iik = (i * ni + i) * ni + k;
                size_t ik = i * ni + k;
                p1[ik] = drand48();
                p_ref[iik] += p1[ik];
            }
            cdt.ret_dataptr(ssn1, p1); p1 = 0;

            double *p2 = cdt.req_dataptr(ssn2);
            for(size_t i = 0; i < ni; i++)
            for(size_t j = 0; j < ni; j++) {
                size_t iji = (i * ni + j) * ni + i;
                size_t ij = i * ni + j;
                p2[ij] = drand48();
                p_ref[iji] += p2[ij];
            }
            cdt.ret_dataptr(ssn2, p2); p2 = 0;

            ct_ref.ret_dataptr(p_ref); p_ref = 0;
        }

        tod_conv_diag_tensor<3>(dt).perform(t);
	compare_ref<3>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_conv_diag_tensor_test::test_1_iijj_ijij_ijjk(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_conv_diag_tensor_test::test_1_iijj_ijij_ijjk(" << ni << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = ni - 1; i2[1] = ni - 1; i2[2] = ni - 1; i2[3] = ni - 1;
        dimensions<4> dims(index_range<4>(i1, i2));
        size_t sz = dims.get_size();

        mask<4> m1100, m1010, m0011, m0101, m0110;
        m1100[0] = true; m1100[1] = true;
        m1010[0] = true; m1010[2] = true;
        m0011[2] = true; m0011[3] = true;
        m0101[1] = true; m0101[3] = true;
        m0110[1] = true; m0110[2] = true;

        diag_tensor_space<4> dts(dims);
        diag_tensor_subspace<4> dtss1(2), dtss2(2), dtss3(1);
        dtss1.set_diag_mask(0, m1100);
        dtss1.set_diag_mask(1, m0011);
        dtss2.set_diag_mask(0, m0101);
        dtss2.set_diag_mask(1, m1010);
        dtss3.set_diag_mask(0, m0110);
        size_t ssn1 = dts.add_subspace(dtss1);
        size_t ssn2 = dts.add_subspace(dtss2);
        size_t ssn3 = dts.add_subspace(dtss3);

        diag_tensor<4, double, allocator_t> dt(dts);
        dense_tensor<4, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<4, double> cdt(dt);
            dense_tensor_wr_ctrl<4, double> ct(t), ct_ref(t_ref);

            double *p = ct.req_dataptr();
            double *p_ref = ct_ref.req_dataptr();
            for(size_t i = 0; i < sz; i++) {
                p[i] = drand48();
                p_ref[i] = 0.0;
            }
            ct.ret_dataptr(p); p = 0;

            double *p1 = cdt.req_dataptr(ssn1);
            for(size_t i = 0; i < ni; i++)
            for(size_t j = 0; j < ni; j++) {
                size_t iijj = ((i * ni + i) * ni + j) * ni + j;
                size_t ij = i * ni + j;
                p1[ij] = drand48();
                p_ref[iijj] += p1[ij];
            }
            cdt.ret_dataptr(ssn1, p1); p1 = 0;

            double *p2 = cdt.req_dataptr(ssn2);
            for(size_t i = 0; i < ni; i++)
            for(size_t j = 0; j < ni; j++) {
                size_t ijij = ((i * ni + j) * ni + i) * ni + j;
                size_t ij = i * ni + j;
                p2[ij] = drand48();
                p_ref[ijij] += p2[ij];
            }
            cdt.ret_dataptr(ssn2, p2); p2 = 0;

            double *p3 = cdt.req_dataptr(ssn3);
            for(size_t i = 0; i < ni; i++)
            for(size_t j = 0; j < ni; j++)
            for(size_t k = 0; k < ni; k++) {
                size_t ijjk = ((i * ni + j) * ni + j) * ni + k;
                size_t ijk = (i * ni + j) * ni + k;
                p3[ijk] = drand48();
                p_ref[ijjk] += p3[ijk];
            }
            cdt.ret_dataptr(ssn3, p3); p3 = 0;

            ct_ref.ret_dataptr(p_ref); p_ref = 0;
        }

        tod_conv_diag_tensor<4>(dt).perform(t);
	compare_ref<4>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

