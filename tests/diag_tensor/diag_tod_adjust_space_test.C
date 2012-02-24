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
}


void diag_tod_adjust_space_test::test_ai_bi(size_t ni)
    throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "diag_tod_adjust_space_test::test_ai_bi(" << ni << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

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

    typedef std_allocator<double> allocator_t;

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

    typedef std_allocator<double> allocator_t;

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


} // namespace libtensor

