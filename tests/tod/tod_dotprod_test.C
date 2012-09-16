#include <cmath> // for fabs()
#include <cstdlib> // for drand48()
#include <sstream>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_dotprod.h>
#include "tod_dotprod_test.h"

namespace libtensor {


void tod_dotprod_test::perform() throw(libtest::test_exception) {

    test_i_i(1);
	test_i_i(4);
    test_i_i(16);
    test_i_i(200);

    test_ij_ij(1, 1);
    test_ij_ij(1, 4);
    test_ij_ij(4, 4);
    test_ij_ij(10, 20);

    test_ij_ji(1, 1);
    test_ij_ji(1, 4);
    test_ij_ji(4, 4);
    test_ij_ji(10, 21);
    test_ij_ji(10, 21);

    test_ijk_ijk(1, 1, 1);
    test_ijk_ijk(1, 1, 4);
    test_ijk_ijk(1, 4, 4);
    test_ijk_ijk(4, 4, 4);
    test_ijk_ijk(10, 11, 12);
    test_ijk_ijk(20, 21, 22);

    test_ijk_ikj(1, 1, 1);
    test_ijk_ikj(1, 1, 4);
    test_ijk_ikj(1, 4, 4);
    test_ijk_ikj(4, 4, 4);
    test_ijk_ikj(10, 11, 12);
    test_ijk_ikj(20, 21, 22);

    test_ijk_jik(1, 1, 1);
    test_ijk_jik(1, 1, 4);
    test_ijk_jik(1, 4, 4);
    test_ijk_jik(4, 4, 4);
    test_ijk_jik(10, 11, 12);
    test_ijk_jik(20, 21, 22);

    test_ijk_jki(1, 1, 1);
    test_ijk_jki(1, 1, 4);
    test_ijk_jki(1, 4, 4);
    test_ijk_jki(4, 4, 4);
    test_ijk_jki(10, 11, 12);
    test_ijk_jki(20, 21, 22);
}


void tod_dotprod_test::test_i_i(size_t ni) throw(libtest::test_exception) {

	std::ostringstream tnss;
    tnss << "tod_dotprod_test::test_i_i(" << ni << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<1> ia1, ia2;
        ia2[0] = ni - 1;
        index<1> ib1, ib2;
        ib2[0] = ni - 1;
        dimensions<1> dima(index_range<1>(ia1, ia2));
        dimensions<1> dimb(index_range<1>(ib1, ib2));
        size_t sza = dima.get_size(), szb = dimb.get_size();

        dense_tensor<1, double, allocator_t> ta(dima);
        dense_tensor<1, double, allocator_t> tb(dimb);

        double c_ref = 0.0;
        {
            dense_tensor_wr_ctrl<1, double> tca(ta);
            dense_tensor_wr_ctrl<1, double> tcb(tb);
            double *dta = tca.req_dataptr();
            double *dtb = tcb.req_dataptr();

            // Fill in random input

            for(size_t i = 0; i < sza; i++) dta[i] = drand48();
            for(size_t i = 0; i < szb; i++) dtb[i] = drand48();

            // Generate reference data

            for(size_t i = 0; i < sza; i++) c_ref += dta[i] * dtb[i];
            tca.ret_dataptr(dta); dta = 0;
            ta.set_immutable();
            tcb.ret_dataptr(dtb); dtb = 0;
            tb.set_immutable();
        }

        // Invoke the operation

        permutation<1> pa, pb;
        double c1 = tod_dotprod<1>(ta, tb).calculate();
        double c2 = tod_dotprod<1>(ta, pa, tb, pb).calculate();

        // Compare against the reference

        if(fabs(c1 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << c1 << " (result), "
                << c_ref << " (reference), " << c1 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c2 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << c2 << " (result), "
                << c_ref << " (reference), " << c2 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_dotprod_test::test_ij_ij(size_t ni, size_t nj)
    throw (libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_dotprod_test::test_ij_ij(" << ni << ", " << nj << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nj - 1;
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = nj - 1;
        dimensions<2> dima(index_range<2>(ia1, ia2));
        dimensions<2> dimb(index_range<2>(ib1, ib2));
        size_t sza = dima.get_size(), szb = dimb.get_size();

        dense_tensor<2, double, allocator_t> ta(dima);
        dense_tensor<2, double, allocator_t> tb(dimb);

        double c_ref = 0.0;
        {
            dense_tensor_wr_ctrl<2, double> tca(ta);
            dense_tensor_wr_ctrl<2, double> tcb(tb);
            double *dta = tca.req_dataptr();
            double *dtb = tcb.req_dataptr();

            // Fill in random input

            for(size_t i = 0; i < sza; i++) dta[i] = drand48();
            for(size_t i = 0; i < szb; i++) dtb[i] = drand48();

            // Generate reference data

            abs_index<2> aia(dima);
            do {
                index<2> ib(aia.get_index());
                abs_index<2> aib(ib, dimb);
                size_t iia = aia.get_abs_index();
                size_t iib = aib.get_abs_index();
                c_ref += dta[iia] * dtb[iib];
            } while(aia.inc());
            tca.ret_dataptr(dta); dta = 0;
            ta.set_immutable();
            tcb.ret_dataptr(dtb); dtb = 0;
            tb.set_immutable();
        }

        // Invoke the operation

        permutation<2> pa, pb;
        double c1 = tod_dotprod<2>(ta, tb).calculate();
        double c2 = tod_dotprod<2>(ta, pa, tb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        double c3 = tod_dotprod<2>(ta, pa, tb, pb).calculate();

        // Compare against the reference

        if(fabs(c1 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << c1 << " (result), "
                << c_ref << " (reference), " << c1 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c2 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << c2 << " (result), "
                << c_ref << " (reference), " << c2 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c3 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (3) doesn't match reference: " << c3 << " (result), "
                << c_ref << " (reference), " << c3 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_dotprod_test::test_ij_ji(size_t ni, size_t nj)
    throw (libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_dotprod_test::test_ij_ji(" << ni << ", " << nj << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nj - 1;
        index<2> ib1, ib2;
        ib2[0] = nj - 1; ib2[1] = ni - 1;
        dimensions<2> dima(index_range<2>(ia1, ia2));
        dimensions<2> dimb(index_range<2>(ib1, ib2));
        size_t sza = dima.get_size(), szb = dimb.get_size();

        dense_tensor<2, double, allocator_t> ta(dima);
        dense_tensor<2, double, allocator_t> tb(dimb);

        double c_ref = 0.0;
        {
            dense_tensor_wr_ctrl<2, double> tca(ta);
            dense_tensor_wr_ctrl<2, double> tcb(tb);
            double *dta = tca.req_dataptr();
            double *dtb = tcb.req_dataptr();

            // Fill in random input

            for(size_t i = 0; i < sza; i++) dta[i] = drand48();
            for(size_t i = 0; i < szb; i++) dtb[i] = drand48();

            // Generate reference data

            permutation<2> p10;
            p10.permute(0, 1);
            abs_index<2> aia(dima);
            do {
                index<2> ib(aia.get_index());
                ib.permute(p10);
                abs_index<2> aib(ib, dimb);
                size_t iia = aia.get_abs_index();
                size_t iib = aib.get_abs_index();
                c_ref += dta[iia] * dtb[iib];
            } while(aia.inc());
            tca.ret_dataptr(dta); dta = 0;
            ta.set_immutable();
            tcb.ret_dataptr(dtb); dtb = 0;
            tb.set_immutable();
        }

        // Invoke the operation

        permutation<2> pa, pb;
        pb.permute(0, 1);
        double c1 = tod_dotprod<2>(ta, pa, tb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        double c2 = tod_dotprod<2>(ta, pa, tb, pb).calculate();

        // Compare against the reference

        if(fabs(c1 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << c1 << " (result), "
                << c_ref << " (reference), " << c1 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c2 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << c2 << " (result), "
                << c_ref << " (reference), " << c2 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_dotprod_test::test_ijk_ijk(size_t ni, size_t nj, size_t nk)
    throw (libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_dotprod_test::test_ijk_ijk(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<3> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nj - 1; ia2[2] = nk - 1;
        index<3> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = nj - 1; ib2[2] = nk - 1;
        dimensions<3> dima(index_range<3>(ia1, ia2));
        dimensions<3> dimb(index_range<3>(ib1, ib2));
        size_t sza = dima.get_size(), szb = dimb.get_size();

        dense_tensor<3, double, allocator_t> ta(dima);
        dense_tensor<3, double, allocator_t> tb(dimb);

        double c_ref = 0.0;
        {
            dense_tensor_wr_ctrl<3, double> tca(ta);
            dense_tensor_wr_ctrl<3, double> tcb(tb);
            double *dta = tca.req_dataptr();
            double *dtb = tcb.req_dataptr();

            // Fill in random input

            for(size_t i = 0; i < sza; i++) dta[i] = drand48();
            for(size_t i = 0; i < szb; i++) dtb[i] = drand48();

            // Generate reference data

            abs_index<3> aia(dima);
            do {
                index<3> ib(aia.get_index());
                abs_index<3> aib(ib, dimb);
                size_t iia = aia.get_abs_index();
                size_t iib = aib.get_abs_index();
                c_ref += dta[iia] * dtb[iib];
            } while(aia.inc());
            tca.ret_dataptr(dta); dta = 0;
            ta.set_immutable();
            tcb.ret_dataptr(dtb); dtb = 0;
            tb.set_immutable();
        }

        // Invoke the operation

        permutation<3> pa, pb;
        double c1 = tod_dotprod<3>(ta, tb).calculate();
        double c2 = tod_dotprod<3>(ta, pa, tb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        double c3 = tod_dotprod<3>(ta, pa, tb, pb).calculate();
        pa.permute(1, 2);
        pb.permute(1, 2);
        double c4 = tod_dotprod<3>(ta, pa, tb, pb).calculate();

        // Compare against the reference

        if(fabs(c1 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << c1 << " (result), "
                << c_ref << " (reference), " << c1 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c2 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << c2 << " (result), "
                << c_ref << " (reference), " << c2 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c3 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (3) doesn't match reference: " << c3 << " (result), "
                << c_ref << " (reference), " << c3 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c4 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (4) doesn't match reference: " << c4 << " (result), "
                << c_ref << " (reference), " << c4 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_dotprod_test::test_ijk_ikj(size_t ni, size_t nj, size_t nk)
    throw (libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_dotprod_test::test_ijk_ikj(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<3> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nj - 1; ia2[2] = nk - 1;
        index<3> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = nk - 1; ib2[2] = nj - 1;
        dimensions<3> dima(index_range<3>(ia1, ia2));
        dimensions<3> dimb(index_range<3>(ib1, ib2));
        size_t sza = dima.get_size(), szb = dimb.get_size();

        dense_tensor<3, double, allocator_t> ta(dima);
        dense_tensor<3, double, allocator_t> tb(dimb);

        double c_ref = 0.0;
        {
            dense_tensor_wr_ctrl<3, double> tca(ta);
            dense_tensor_wr_ctrl<3, double> tcb(tb);
            double *dta = tca.req_dataptr();
            double *dtb = tcb.req_dataptr();

            // Fill in random input

            for(size_t i = 0; i < sza; i++) dta[i] = drand48();
            for(size_t i = 0; i < szb; i++) dtb[i] = drand48();

            // Generate reference data

            permutation<3> p021;
            p021.permute(1, 2);
            abs_index<3> aia(dima);
            do {
                index<3> ib(aia.get_index());
                ib.permute(p021);
                abs_index<3> aib(ib, dimb);
                size_t iia = aia.get_abs_index();
                size_t iib = aib.get_abs_index();
                c_ref += dta[iia] * dtb[iib];
            } while(aia.inc());
            tca.ret_dataptr(dta); dta = 0;
            ta.set_immutable();
            tcb.ret_dataptr(dtb); dtb = 0;
            tb.set_immutable();
        }

        // Invoke the operation

        permutation<3> pa, pb;
        pb.permute(1, 2);
        double c1 = tod_dotprod<3>(ta, pa, tb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        double c2 = tod_dotprod<3>(ta, pa, tb, pb).calculate();
        pa.permute(1, 2);
        pb.permute(1, 2);
        double c3 = tod_dotprod<3>(ta, pa, tb, pb).calculate();
        pa.permute(0, 2);
        pb.permute(0, 2);
        double c4 = tod_dotprod<3>(ta, pa, tb, pb).calculate();

        // Compare against the reference

        if(fabs(c1 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << c1 << " (result), "
                << c_ref << " (reference), " << c1 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c2 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << c2 << " (result), "
                << c_ref << " (reference), " << c2 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c3 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (3) doesn't match reference: " << c3 << " (result), "
                << c_ref << " (reference), " << c3 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c4 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (4) doesn't match reference: " << c4 << " (result), "
                << c_ref << " (reference), " << c4 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_dotprod_test::test_ijk_jik(size_t ni, size_t nj, size_t nk)
    throw (libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_dotprod_test::test_ijk_jik(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<3> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nj - 1; ia2[2] = nk - 1;
        index<3> ib1, ib2;
        ib2[0] = nj - 1; ib2[1] = ni - 1; ib2[2] = nk - 1;
        dimensions<3> dima(index_range<3>(ia1, ia2));
        dimensions<3> dimb(index_range<3>(ib1, ib2));
        size_t sza = dima.get_size(), szb = dimb.get_size();

        dense_tensor<3, double, allocator_t> ta(dima);
        dense_tensor<3, double, allocator_t> tb(dimb);

        double c_ref = 0.0;
        {
            dense_tensor_wr_ctrl<3, double> tca(ta);
            dense_tensor_wr_ctrl<3, double> tcb(tb);
            double *dta = tca.req_dataptr();
            double *dtb = tcb.req_dataptr();

            // Fill in random input

            for(size_t i = 0; i < sza; i++) dta[i] = drand48();
            for(size_t i = 0; i < szb; i++) dtb[i] = drand48();

            // Generate reference data

            permutation<3> p102;
            p102.permute(0, 1);
            abs_index<3> aia(dima);
            do {
                index<3> ib(aia.get_index());
                ib.permute(p102);
                abs_index<3> aib(ib, dimb);
                size_t iia = aia.get_abs_index();
                size_t iib = aib.get_abs_index();
                c_ref += dta[iia] * dtb[iib];
            } while(aia.inc());
            tca.ret_dataptr(dta); dta = 0;
            ta.set_immutable();
            tcb.ret_dataptr(dtb); dtb = 0;
            tb.set_immutable();
        }

        // Invoke the operation

        permutation<3> pa, pb;
        pb.permute(0, 1);
        double c1 = tod_dotprod<3>(ta, pa, tb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        double c2 = tod_dotprod<3>(ta, pa, tb, pb).calculate();
        pa.permute(1, 2);
        pb.permute(1, 2);
        double c3 = tod_dotprod<3>(ta, pa, tb, pb).calculate();
        pa.permute(0, 2);
        pb.permute(0, 2);
        double c4 = tod_dotprod<3>(ta, pa, tb, pb).calculate();

        // Compare against the reference

        if(fabs(c1 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << c1 << " (result), "
                << c_ref << " (reference), " << c1 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c2 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << c2 << " (result), "
                << c_ref << " (reference), " << c2 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c3 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (3) doesn't match reference: " << c3 << " (result), "
                << c_ref << " (reference), " << c3 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c4 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (4) doesn't match reference: " << c4 << " (result), "
                << c_ref << " (reference), " << c4 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


void tod_dotprod_test::test_ijk_jki(size_t ni, size_t nj, size_t nk)
    throw (libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "tod_dotprod_test::test_ijk_jki(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;

    try {

        index<3> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nj - 1; ia2[2] = nk - 1;
        index<3> ib1, ib2;
        ib2[0] = nj - 1; ib2[1] = nk - 1; ib2[2] = ni - 1;
        dimensions<3> dima(index_range<3>(ia1, ia2));
        dimensions<3> dimb(index_range<3>(ib1, ib2));
        size_t sza = dima.get_size(), szb = dimb.get_size();

        dense_tensor<3, double, allocator_t> ta(dima);
        dense_tensor<3, double, allocator_t> tb(dimb);

        double c_ref = 0.0;
        {
            dense_tensor_wr_ctrl<3, double> tca(ta);
            dense_tensor_wr_ctrl<3, double> tcb(tb);
            double *dta = tca.req_dataptr();
            double *dtb = tcb.req_dataptr();

            // Fill in random input

            for(size_t i = 0; i < sza; i++) dta[i] = drand48();
            for(size_t i = 0; i < szb; i++) dtb[i] = drand48();

            // Generate reference data

            permutation<3> p120;
            p120.permute(0, 1).permute(1, 2);
            abs_index<3> aia(dima);
            do {
                index<3> ib(aia.get_index());
                ib.permute(p120);
                abs_index<3> aib(ib, dimb);
                size_t iia = aia.get_abs_index();
                size_t iib = aib.get_abs_index();
                c_ref += dta[iia] * dtb[iib];
            } while(aia.inc());
            tca.ret_dataptr(dta); dta = 0;
            ta.set_immutable();
            tcb.ret_dataptr(dtb); dtb = 0;
            tb.set_immutable();
        }

        // Invoke the operation

        permutation<3> pa, pb;
        pb.permute(1, 2).permute(0, 1); // jki -> ijk
        double c1 = tod_dotprod<3>(ta, pa, tb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        double c2 = tod_dotprod<3>(ta, pa, tb, pb).calculate();
        pa.permute(1, 2);
        pb.permute(1, 2);
        double c3 = tod_dotprod<3>(ta, pa, tb, pb).calculate();
        pa.permute(0, 2);
        pb.permute(0, 2);
        double c4 = tod_dotprod<3>(ta, pa, tb, pb).calculate();

        // Compare against the reference

        if(fabs(c1 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << c1 << " (result), "
                << c_ref << " (reference), " << c1 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c2 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << c2 << " (result), "
                << c_ref << " (reference), " << c2 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c3 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (3) doesn't match reference: " << c3 << " (result), "
                << c_ref << " (reference), " << c3 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(fabs(c4 - c_ref) > 1e-14 * fabs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (4) doesn't match reference: " << c4 << " (result), "
                << c_ref << " (reference), " << c4 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
