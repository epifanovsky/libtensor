#include <cstdlib>
#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/diag_tensor/diag_tensor.h>
#include <libtensor/diag_tensor/diag_tensor_ctrl.h>
#include <libtensor/diag_tensor/diag_tod_adjust_space.h>
#include <libtensor/diag_tensor/tod_conv_diag_tensor.h>
#include "../compare_ref.h"
#include "diag_tod_adjust_space_test.h"

namespace libtensor {


void diag_tod_adjust_space_test::perform() throw(libtest::test_exception) {

    test_ai_bi(1);
    test_ai_bi(4);
    test_ai_bi(21);

    test_ai_ai_bi(1);
    test_ai_ai_bi(4);
    test_ai_ai_bi(21);

    test_aijk_biij_biii(1);
    test_aijk_biij_biii(4);
    test_aijk_biij_biii(21);

    test_aiii_aijk_biii_biij(1);
    test_aiii_aijk_biii_biij(4);
    test_aiii_aijk_biii_biij(21);

    test_aijkl_biiii(1);
    test_aijkl_biiii(4);
    test_aijkl_biiii(21);

    test_aijkl_biiii_bijkl(1);
    test_aijkl_biiii_bijkl(4);
    test_aijkl_biiii_bijkl(21);

    test_aiiii_biijj_bijij(1);
    test_aiiii_biijj_bijij(4);
    test_aiiii_biijj_bijij(21);

    test_aiiii_aijij_biijj_bijij(1);
    test_aiiii_aijij_biijj_bijij(4);
    test_aiiii_aijij_biijj_bijij(21);

    test_aiijj_biiij(1);
    test_aiijj_biiij(4);
    test_aiijj_biiij(21);
}


void diag_tod_adjust_space_test::test_ai_bi(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "diag_tod_adjust_space_test::test_ai_bi(" << ni << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<1> i1, i2;
        i2[0] = ni - 1;
        dimensions<1> dims(index_range<1>(i1, i2));
        size_t sz = dims.get_size();

        mask<1> m1;
        m1[0] = true;

        diag_tensor_space<1> dtsa(dims), dtsb(dims);
        diag_tensor_subspace<1> dtssa1(0), dtssb1(1);
        dtssb1.set_diag_mask(0, m1);
        size_t ssn1 = dtsa.add_subspace(dtssa1);
        dtsb.add_subspace(dtssb1);

        diag_tensor<1, double, allocator_t> dt(dtsa);
        dense_tensor<1, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<1, double> cdt(dt);
            dense_tensor_wr_ctrl<1, double> ct_ref(t_ref);
            double *pa1 = cdt.req_dataptr(ssn1);
            double *pb = ct_ref.req_dataptr();
            for(size_t i = 0; i < sz; i++) pa1[i] = pb[i] = drand48();
            cdt.ret_dataptr(ssn1, pa1); pa1 = 0;
            ct_ref.ret_dataptr(pb); pb = 0;
        }

        diag_tod_adjust_space<1>(dtsb).perform(dt);
        tod_conv_diag_tensor<1>(dt).perform(t);

        if(dt.get_space().get_nsubspaces() != 1) {
            fail_test(tn.c_str(), __FILE__, __LINE__, "nsubspaces != 1");
        }

        compare_ref<1>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void diag_tod_adjust_space_test::test_ai_ai_bi(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "diag_tod_adjust_space_test::test_ai_ai_bi(" << ni << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<1> i1, i2;
        i2[0] = ni - 1;
        dimensions<1> dims(index_range<1>(i1, i2));
        size_t sz = dims.get_size();

        mask<1> m1;
        m1[0] = true;

        diag_tensor_space<1> dtsa(dims), dtsb(dims);
        diag_tensor_subspace<1> dtssa1(0), dtssa2(1), dtssb1(1);
        dtssa2.set_diag_mask(0, m1);
        dtssb1.set_diag_mask(0, m1);
        size_t ssn1 = dtsa.add_subspace(dtssa1);
        size_t ssn2 = dtsa.add_subspace(dtssa2);
        dtsb.add_subspace(dtssb1);

        diag_tensor<1, double, allocator_t> dt(dtsa);
        dense_tensor<1, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<1, double> cdt(dt);
            dense_tensor_wr_ctrl<1, double> ct_ref(t_ref);
            double *pa1 = cdt.req_dataptr(ssn1);
            double *pa2 = cdt.req_dataptr(ssn2);
            double *pb = ct_ref.req_dataptr();
            for(size_t i = 0; i < sz; i++) {
                double d1 = pa1[i] = drand48();
                double d2 = pa2[i] = -drand48();
                pb[i] = d1 + d2;
            }
            cdt.ret_dataptr(ssn1, pa1); pa1 = 0;
            cdt.ret_dataptr(ssn2, pa2); pa2 = 0;
            ct_ref.ret_dataptr(pb); pb = 0;
        }

        diag_tod_adjust_space<1>(dtsb).perform(dt);
        tod_conv_diag_tensor<1>(dt).perform(t);

        if(dt.get_space().get_nsubspaces() != 1) {
            fail_test(tn.c_str(), __FILE__, __LINE__, "nsubspaces != 1");
        }

        compare_ref<1>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void diag_tod_adjust_space_test::test_aijk_biij_biii(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "diag_tod_adjust_space_test::test_aijk_biij_biii(" << ni << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<3> i1, i2;
        i2[0] = ni - 1; i2[1] = ni - 1; i2[2] = ni - 1;
        dimensions<3> dims(index_range<3>(i1, i2));

        mask<3> m110, m001, m111;
        m110[0] = true; m110[1] = true; m001[2] = true;
        m111[0] = true; m111[1] = true; m111[2] = true;

        diag_tensor_space<3> dtsa(dims), dtsb(dims);
        diag_tensor_subspace<3> dtssa1(0), dtssb1(2), dtssb2(1);
        dtssb1.set_diag_mask(0, m110);
        dtssb1.set_diag_mask(1, m001);
        dtssb2.set_diag_mask(0, m111);
        size_t ssn1 = dtsa.add_subspace(dtssa1);
        dtsb.add_subspace(dtssb1);
        dtsb.add_subspace(dtssb2);

        diag_tensor<3, double, allocator_t> dt(dtsa);
        dense_tensor<3, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<3, double> cdt(dt);
            dense_tensor_wr_ctrl<3, double> ct_ref(t_ref);
            double *pa1 = cdt.req_dataptr(ssn1);
            double *pb = ct_ref.req_dataptr();
            for(size_t i = 0; i < ni; i++)
            for(size_t j = 0; j < ni; j++)
            for(size_t k = 0; k < ni; k++) {
                size_t ijk = (i * ni + j) * ni + k;
                pa1[ijk] = drand48();
                pb[ijk] = 0.0;
            }
            //  iii part is included here as well
            for(size_t i = 0; i < ni; i++)
            for(size_t j = 0; j < ni; j++) {
                size_t iij = (i * ni + i) * ni + j;
                pb[iij] = pa1[iij];
            }
            cdt.ret_dataptr(ssn1, pa1); pa1 = 0;
            ct_ref.ret_dataptr(pb); pb = 0;
        }

        diag_tod_adjust_space<3>(dtsb).perform(dt);
        tod_conv_diag_tensor<3>(dt).perform(t);

        if(dt.get_space().get_nsubspaces() != 2) {
            fail_test(tn.c_str(), __FILE__, __LINE__, "nsubspaces != 2");
        }

        compare_ref<3>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void diag_tod_adjust_space_test::test_aiii_aijk_biii_biij(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "diag_tod_adjust_space_test::test_aiii_aijk_biii_biij("
        << ni << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<3> i1, i2;
        i2[0] = ni - 1; i2[1] = ni - 1; i2[2] = ni - 1;
        dimensions<3> dims(index_range<3>(i1, i2));

        mask<3> m110, m001, m111;
        m110[0] = true; m110[1] = true; m001[2] = true;
        m111[0] = true; m111[1] = true; m111[2] = true;

        diag_tensor_space<3> dtsa(dims), dtsb(dims);
        diag_tensor_subspace<3> dtssa1(1), dtssa2(0), dtssb1(1), dtssb2(2);
        dtssa1.set_diag_mask(0, m111);
        dtssb1.set_diag_mask(0, m111);
        dtssb2.set_diag_mask(0, m110);
        dtssb2.set_diag_mask(1, m001);
        size_t ssn1 = dtsa.add_subspace(dtssa1);
        size_t ssn2 = dtsa.add_subspace(dtssa2);
        dtsb.add_subspace(dtssb1);
        dtsb.add_subspace(dtssb2);

        diag_tensor<3, double, allocator_t> dt(dtsa);
        dense_tensor<3, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<3, double> cdt(dt);
            dense_tensor_wr_ctrl<3, double> ct_ref(t_ref);
            double *pa1 = cdt.req_dataptr(ssn1);
            double *pa2 = cdt.req_dataptr(ssn2);
            double *pb = ct_ref.req_dataptr();
            for(size_t i = 0; i < ni; i++)
            for(size_t j = 0; j < ni; j++)
            for(size_t k = 0; k < ni; k++) {
                size_t ijk = (i * ni + j) * ni + k;
                pa1[i] = drand48();
                pa2[ijk] = drand48();
                pb[ijk] = 0.0;
            }
            for(size_t i = 0; i < ni; i++) {
                size_t iii = (i * ni + i) * ni + i;
                pb[iii] += pa1[i];
            }
            for(size_t i = 0; i < ni; i++)
            for(size_t j = 0; j < ni; j++) {
                size_t iij = (i * ni + i) * ni + j;
                pb[iij] += pa2[iij];
            }
            cdt.ret_dataptr(ssn1, pa1); pa1 = 0;
            cdt.ret_dataptr(ssn2, pa2); pa2 = 0;
            ct_ref.ret_dataptr(pb); pb = 0;
        }

        diag_tod_adjust_space<3>(dtsb).perform(dt);
        tod_conv_diag_tensor<3>(dt).perform(t);

        if(dt.get_space().get_nsubspaces() != 2) {
            fail_test(tn.c_str(), __FILE__, __LINE__, "nsubspaces != 2");
        }

        compare_ref<3>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void diag_tod_adjust_space_test::test_aijkl_biiii(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "diag_tod_adjust_space_test::test_aijkl_biiii(" << ni << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = ni - 1; i2[1] = ni - 1; i2[2] = ni - 1; i2[3] = ni - 1;
        dimensions<4> dims(index_range<4>(i1, i2));

        mask<4> m1111;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

        diag_tensor_space<4> dtsa(dims), dtsb(dims);
        diag_tensor_subspace<4> dtssa1(0), dtssb1(1);
        dtssb1.set_diag_mask(0, m1111);
        size_t ssn1 = dtsa.add_subspace(dtssa1);
        dtsb.add_subspace(dtssb1);

        diag_tensor<4, double, allocator_t> dt(dtsa);
        dense_tensor<4, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<4, double> cdt(dt);
            dense_tensor_wr_ctrl<4, double> ct_ref(t_ref);
            double *pa1 = cdt.req_dataptr(ssn1);
            double *pb = ct_ref.req_dataptr();
            for(size_t i = 0; i < ni; i++)
            for(size_t j = 0; j < ni; j++)
            for(size_t k = 0; k < ni; k++)
            for(size_t l = 0; l < ni; l++) {
                size_t ijkl = ((i * ni + j) * ni + k) * ni + l;
                pa1[ijkl] = drand48();
                pb[ijkl] = 0.0;
            }
            for(size_t i = 0; i < ni; i++) {
                size_t iiii = ((i * ni + i) * ni + i) * ni + i;
                pb[iiii] = pa1[iiii];
            }
            cdt.ret_dataptr(ssn1, pa1); pa1 = 0;
            ct_ref.ret_dataptr(pb); pb = 0;
        }

        diag_tod_adjust_space<4>(dtsb).perform(dt);
        tod_conv_diag_tensor<4>(dt).perform(t);

        if(dt.get_space().get_nsubspaces() != 1) {
            fail_test(tn.c_str(), __FILE__, __LINE__, "nsubspaces != 1");
        }

        compare_ref<4>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void diag_tod_adjust_space_test::test_aijkl_biiii_bijkl(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "diag_tod_adjust_space_test::test_aijkl_biiii_bijkl(" << ni << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = ni - 1; i2[1] = ni - 1; i2[2] = ni - 1; i2[3] = ni - 1;
        dimensions<4> dims(index_range<4>(i1, i2));

        mask<4> m1111;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

        diag_tensor_space<4> dtsa(dims), dtsb(dims);
        diag_tensor_subspace<4> dtssa1(0), dtssb1(1), dtssb2(0);
        dtssb1.set_diag_mask(0, m1111);
        size_t ssn1 = dtsa.add_subspace(dtssa1);
        dtsb.add_subspace(dtssb1);
        dtsb.add_subspace(dtssb2);

        diag_tensor<4, double, allocator_t> dt(dtsa);
        dense_tensor<4, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<4, double> cdt(dt);
            dense_tensor_wr_ctrl<4, double> ct_ref(t_ref);
            double *pa1 = cdt.req_dataptr(ssn1);
            double *pb = ct_ref.req_dataptr();
            for(size_t i = 0; i < ni; i++)
            for(size_t j = 0; j < ni; j++)
            for(size_t k = 0; k < ni; k++)
            for(size_t l = 0; l < ni; l++) {
                size_t ijkl = ((i * ni + j) * ni + k) * ni + l;
                pb[ijkl] = pa1[ijkl] = drand48();
            }
            cdt.ret_dataptr(ssn1, pa1); pa1 = 0;
            ct_ref.ret_dataptr(pb); pb = 0;
        }

        diag_tod_adjust_space<4>(dtsb).perform(dt);
        tod_conv_diag_tensor<4>(dt).perform(t);

        if(dt.get_space().get_nsubspaces() != 2) {
            fail_test(tn.c_str(), __FILE__, __LINE__, "nsubspaces != 2");
        }

        //  Need to check that b_iiii part is zero

        compare_ref<4>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void diag_tod_adjust_space_test::test_aiiii_biijj_bijij(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "diag_tod_adjust_space_test::test_aiiii_biijj_bijij(" << ni << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = ni - 1; i2[1] = ni - 1; i2[2] = ni - 1; i2[3] = ni - 1;
        dimensions<4> dims(index_range<4>(i1, i2));
        size_t sz = dims.get_size();

        mask<4> m0011, m0101, m1010, m1100, m1111;
        m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
        m1010[0] = true; m0101[1] = true; m1010[2] = true; m0101[3] = true;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

        diag_tensor_space<4> dtsa(dims), dtsb(dims);
        diag_tensor_subspace<4> dtssa1(1), dtssb1(2), dtssb2(2);
        dtssa1.set_diag_mask(0, m1111);
        dtssb1.set_diag_mask(0, m0011);
        dtssb1.set_diag_mask(1, m1100);
        dtssb2.set_diag_mask(0, m1010);
        dtssb2.set_diag_mask(1, m0101);
        size_t ssn1 = dtsa.add_subspace(dtssa1);
        dtsb.add_subspace(dtssb1);
        dtsb.add_subspace(dtssb2);

        diag_tensor<4, double, allocator_t> dt(dtsa);
        dense_tensor<4, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<4, double> cdt(dt);
            dense_tensor_wr_ctrl<4, double> ct_ref(t_ref);
            double *pa1 = cdt.req_dataptr(ssn1);
            double *pb = ct_ref.req_dataptr();
            for(size_t i = 0; i < sz; i++) pb[i] = 0.0;
            for(size_t i = 0; i < ni; i++) {
                size_t iiii = ((i * ni + i) * ni + i) * ni + i;
                pa1[i] = drand48();
                pb[iiii] += pa1[i];
            }
            cdt.ret_dataptr(ssn1, pa1); pa1 = 0;
            ct_ref.ret_dataptr(pb); pb = 0;
        }

        diag_tod_adjust_space<4>(dtsb).perform(dt);
        tod_conv_diag_tensor<4>(dt).perform(t);

        if(dt.get_space().get_nsubspaces() != 2) {
            fail_test(tn.c_str(), __FILE__, __LINE__, "nsubspaces != 2");
        }

        compare_ref<4>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void diag_tod_adjust_space_test::test_aiiii_aijij_biijj_bijij(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "diag_tod_adjust_space_test::test_aiiii_aijij_biijj_bijij("
        << ni << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = ni - 1; i2[1] = ni - 1; i2[2] = ni - 1; i2[3] = ni - 1;
        dimensions<4> dims(index_range<4>(i1, i2));
        size_t sz = dims.get_size();

        mask<4> m0011, m0101, m1010, m1100, m1111;
        m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
        m1010[0] = true; m0101[1] = true; m1010[2] = true; m0101[3] = true;
        m1111[0] = true; m1111[1] = true; m1111[2] = true; m1111[3] = true;

        diag_tensor_space<4> dtsa(dims), dtsb(dims);
        diag_tensor_subspace<4> dtssa1(1), dtssa2(2), dtssb1(2), dtssb2(2);
        dtssa1.set_diag_mask(0, m1111);
        dtssa2.set_diag_mask(0, m1010);
        dtssa2.set_diag_mask(1, m0101);
        dtssb1.set_diag_mask(0, m0011);
        dtssb1.set_diag_mask(1, m1100);
        dtssb2.set_diag_mask(0, m1010);
        dtssb2.set_diag_mask(1, m0101);
        size_t ssn1 = dtsa.add_subspace(dtssa1);
        size_t ssn2 = dtsa.add_subspace(dtssa2);
        dtsb.add_subspace(dtssb1);
        dtsb.add_subspace(dtssb2);

        diag_tensor<4, double, allocator_t> dt(dtsa);
        dense_tensor<4, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<4, double> cdt(dt);
            dense_tensor_wr_ctrl<4, double> ct_ref(t_ref);
            double *pa1 = cdt.req_dataptr(ssn1);
            double *pa2 = cdt.req_dataptr(ssn2);
            double *pb = ct_ref.req_dataptr();
            for(size_t i = 0; i < sz; i++) pb[i] = 0.0;
            for(size_t i = 0; i < ni; i++) {
                size_t iiii = ((i * ni + i) * ni + i) * ni + i;
                pa1[i] = drand48();
                pb[iiii] += pa1[i];
            }
            for(size_t i = 0; i < ni; i++)
            for(size_t j = 0; j < ni; j++) {
                size_t ij = i * ni + j;
                size_t ijij = ((i * ni + j) * ni + i) * ni + j;
                pa2[ij] = drand48();
                pb[ijij] += pa2[ij];
            }
            cdt.ret_dataptr(ssn1, pa1); pa1 = 0;
            cdt.ret_dataptr(ssn2, pa2); pa2 = 0;
            ct_ref.ret_dataptr(pb); pb = 0;
        }

        diag_tod_adjust_space<4>(dtsb).perform(dt);
        tod_conv_diag_tensor<4>(dt).perform(t);

        if(dt.get_space().get_nsubspaces() != 2) {
            fail_test(tn.c_str(), __FILE__, __LINE__, "nsubspaces != 2");
        }

        compare_ref<4>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void diag_tod_adjust_space_test::test_aiijj_biiij(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "diag_tod_adjust_space_test::test_aiijj_biiij(" << ni << ")";
    std::string tn = tnss.str();

    typedef allocator<double> allocator_t;

    try {

        index<4> i1, i2;
        i2[0] = ni - 1; i2[1] = ni - 1; i2[2] = ni - 1; i2[3] = ni - 1;
        dimensions<4> dims(index_range<4>(i1, i2));
        size_t sz = dims.get_size();

        mask<4> m0011, m1100, m1110;
        m1100[0] = true; m1100[1] = true; m0011[2] = true; m0011[3] = true;
        m1110[0] = true; m1110[1] = true; m1110[2] = true;

        diag_tensor_space<4> dtsa(dims), dtsb(dims);
        diag_tensor_subspace<4> dtssa1(2), dtssb1(1);
        dtssa1.set_diag_mask(0, m0011);
        dtssa1.set_diag_mask(1, m1100);
        dtssb1.set_diag_mask(0, m1110);
        size_t ssn1 = dtsa.add_subspace(dtssa1);
        dtsb.add_subspace(dtssb1);

        diag_tensor<4, double, allocator_t> dt(dtsa);
        dense_tensor<4, double, allocator_t> t(dims), t_ref(dims);

        {
            diag_tensor_wr_ctrl<4, double> cdt(dt);
            dense_tensor_wr_ctrl<4, double> ct_ref(t_ref);
            double *pa1 = cdt.req_dataptr(ssn1);
            double *pb = ct_ref.req_dataptr();
            for(size_t i = 0; i < sz; i++) pb[i] = 0.0;
            for(size_t i = 0; i < ni; i++)
            for(size_t j = 0; j < ni; j++) {
                size_t ij = i * ni + j;
                pa1[ij] = drand48();
            }
            for(size_t i = 0; i < ni; i++) {
                size_t ii = i * ni + i;
                size_t iiii = ((i * ni + i) * ni + i) * ni + i;
                pb[iiii] = pa1[ii];
            }
            cdt.ret_dataptr(ssn1, pa1); pa1 = 0;
            ct_ref.ret_dataptr(pb); pb = 0;
        }

        diag_tod_adjust_space<4>(dtsb).perform(dt);
        tod_conv_diag_tensor<4>(dt).perform(t);

        if(dt.get_space().get_nsubspaces() != 1) {
            fail_test(tn.c_str(), __FILE__, __LINE__, "nsubspaces != 1");
        }

        compare_ref<4>::compare(tn.c_str(), t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor

