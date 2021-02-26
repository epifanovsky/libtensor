#include <cmath> // for std::abs()
#include <cstdlib> // for drand48()
#include <sstream>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_dotprod.h>
#include "to_dotprod_test.h"

namespace libtensor {

void to_dotprod_test::perform() throw(libtest::test_exception) {
    std::cout << "Testing to_dotprod_test_x<double>  ";
    to_dotprod_test_x<double> t_double;
    t_double.perform();
    std::cout << "Testing to_dotprod_test_x<float>  ";
    to_dotprod_test_x<float> t_float;
    t_float.perform();
}

template<>
const double to_dotprod_test_x<double>::k_thresh = 7e-14;

template<>
const float to_dotprod_test_x<float>::k_thresh = 1e-5;

template <typename T>
void to_dotprod_test_x<T>::perform() throw(libtest::test_exception) {

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


template <typename T>
void to_dotprod_test_x<T>::test_i_i(size_t ni) throw(libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_dotprod_test_x<T>::test_i_i(" << ni << ")";
    std::string tn = tnss.str();

    typedef allocator<T> allocator_t;

    try {

        index<1> ia1, ia2;
        ia2[0] = ni - 1;
        index<1> ib1, ib2;
        ib2[0] = ni - 1;
        dimensions<1> dima(index_range<1>(ia1, ia2));
        dimensions<1> dimb(index_range<1>(ib1, ib2));
        size_t sza = dima.get_size(), szb = dimb.get_size();

        dense_tensor<1, T, allocator_t> ta(dima);
        dense_tensor<1, T, allocator_t> tb(dimb);

        T c_ref = 0.0;
        {
            dense_tensor_wr_ctrl<1, T> tca(ta);
            dense_tensor_wr_ctrl<1, T> tcb(tb);
            T *dta = tca.req_dataptr();
            T *dtb = tcb.req_dataptr();

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
        T c1 = to_dotprod<1, T>(ta, tb).calculate();
        T c2 = to_dotprod<1, T>(ta, pa, tb, pb).calculate();

        // Compare against the reference

        if(std::abs(c1 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << c1 << " (result), "
                << c_ref << " (reference), " << c1 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c2 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << c2 << " (result), "
                << c_ref << " (reference), " << c2 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


template <typename T>
void to_dotprod_test_x<T>::test_ij_ij(size_t ni, size_t nj)
    throw (libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_dotprod_test_x<T>::test_ij_ij(" << ni << ", " << nj << ")";
    std::string tn = tnss.str();

    typedef allocator<T> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nj - 1;
        index<2> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = nj - 1;
        dimensions<2> dima(index_range<2>(ia1, ia2));
        dimensions<2> dimb(index_range<2>(ib1, ib2));
        size_t sza = dima.get_size(), szb = dimb.get_size();

        dense_tensor<2, T, allocator_t> ta(dima);
        dense_tensor<2, T, allocator_t> tb(dimb);

        T c_ref = 0.0;
        {
            dense_tensor_wr_ctrl<2, T> tca(ta);
            dense_tensor_wr_ctrl<2, T> tcb(tb);
            T *dta = tca.req_dataptr();
            T *dtb = tcb.req_dataptr();

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
        T c1 = to_dotprod<2, T>(ta, tb).calculate();
        T c2 = to_dotprod<2, T>(ta, pa, tb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        T c3 = to_dotprod<2, T>(ta, pa, tb, pb).calculate();

        // Compare against the reference

        if(std::abs(c1 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << c1 << " (result), "
                << c_ref << " (reference), " << c1 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c2 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << c2 << " (result), "
                << c_ref << " (reference), " << c2 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c3 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (3) doesn't match reference: " << c3 << " (result), "
                << c_ref << " (reference), " << c3 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


template <typename T>
void to_dotprod_test_x<T>::test_ij_ji(size_t ni, size_t nj)
    throw (libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_dotprod_test_x<T>::test_ij_ji(" << ni << ", " << nj << ")";
    std::string tn = tnss.str();

    typedef allocator<T> allocator_t;

    try {

        index<2> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nj - 1;
        index<2> ib1, ib2;
        ib2[0] = nj - 1; ib2[1] = ni - 1;
        dimensions<2> dima(index_range<2>(ia1, ia2));
        dimensions<2> dimb(index_range<2>(ib1, ib2));
        size_t sza = dima.get_size(), szb = dimb.get_size();

        dense_tensor<2, T, allocator_t> ta(dima);
        dense_tensor<2, T, allocator_t> tb(dimb);

        T c_ref = 0.0;
        {
            dense_tensor_wr_ctrl<2, T> tca(ta);
            dense_tensor_wr_ctrl<2, T> tcb(tb);
            T *dta = tca.req_dataptr();
            T *dtb = tcb.req_dataptr();

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
        T c1 = to_dotprod<2, T>(ta, pa, tb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        T c2 = to_dotprod<2, T>(ta, pa, tb, pb).calculate();

        // Compare against the reference

        if(std::abs(c1 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << c1 << " (result), "
                << c_ref << " (reference), " << c1 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c2 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << c2 << " (result), "
                << c_ref << " (reference), " << c2 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


template <typename T>
void to_dotprod_test_x<T>::test_ijk_ijk(size_t ni, size_t nj, size_t nk)
    throw (libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_dotprod_test_x<T>::test_ijk_ijk(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    typedef allocator<T> allocator_t;

    try {

        index<3> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nj - 1; ia2[2] = nk - 1;
        index<3> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = nj - 1; ib2[2] = nk - 1;
        dimensions<3> dima(index_range<3>(ia1, ia2));
        dimensions<3> dimb(index_range<3>(ib1, ib2));
        size_t sza = dima.get_size(), szb = dimb.get_size();

        dense_tensor<3, T, allocator_t> ta(dima);
        dense_tensor<3, T, allocator_t> tb(dimb);

        T c_ref = 0.0;
        {
            dense_tensor_wr_ctrl<3, T> tca(ta);
            dense_tensor_wr_ctrl<3, T> tcb(tb);
            T *dta = tca.req_dataptr();
            T *dtb = tcb.req_dataptr();

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
        T c1 = to_dotprod<3, T>(ta, tb).calculate();
        T c2 = to_dotprod<3, T>(ta, pa, tb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        T c3 = to_dotprod<3, T>(ta, pa, tb, pb).calculate();
        pa.permute(1, 2);
        pb.permute(1, 2);
        T c4 = to_dotprod<3, T>(ta, pa, tb, pb).calculate();

        // Compare against the reference

        if(std::abs(c1 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << c1 << " (result), "
                << c_ref << " (reference), " << c1 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c2 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << c2 << " (result), "
                << c_ref << " (reference), " << c2 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c3 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (3) doesn't match reference: " << c3 << " (result), "
                << c_ref << " (reference), " << c3 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c4 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (4) doesn't match reference: " << c4 << " (result), "
                << c_ref << " (reference), " << c4 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


template <typename T>
void to_dotprod_test_x<T>::test_ijk_ikj(size_t ni, size_t nj, size_t nk)
    throw (libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_dotprod_test_x<T>::test_ijk_ikj(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    typedef allocator<T> allocator_t;

    try {

        index<3> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nj - 1; ia2[2] = nk - 1;
        index<3> ib1, ib2;
        ib2[0] = ni - 1; ib2[1] = nk - 1; ib2[2] = nj - 1;
        dimensions<3> dima(index_range<3>(ia1, ia2));
        dimensions<3> dimb(index_range<3>(ib1, ib2));
        size_t sza = dima.get_size(), szb = dimb.get_size();

        dense_tensor<3, T, allocator_t> ta(dima);
        dense_tensor<3, T, allocator_t> tb(dimb);

        T c_ref = 0.0;
        {
            dense_tensor_wr_ctrl<3, T> tca(ta);
            dense_tensor_wr_ctrl<3, T> tcb(tb);
            T *dta = tca.req_dataptr();
            T *dtb = tcb.req_dataptr();

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
        T c1 = to_dotprod<3, T>(ta, pa, tb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        T c2 = to_dotprod<3, T>(ta, pa, tb, pb).calculate();
        pa.permute(1, 2);
        pb.permute(1, 2);
        T c3 = to_dotprod<3, T>(ta, pa, tb, pb).calculate();
        pa.permute(0, 2);
        pb.permute(0, 2);
        T c4 = to_dotprod<3, T>(ta, pa, tb, pb).calculate();

        // Compare against the reference

        if(std::abs(c1 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << c1 << " (result), "
                << c_ref << " (reference), " << c1 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c2 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << c2 << " (result), "
                << c_ref << " (reference), " << c2 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c3 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (3) doesn't match reference: " << c3 << " (result), "
                << c_ref << " (reference), " << c3 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c4 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (4) doesn't match reference: " << c4 << " (result), "
                << c_ref << " (reference), " << c4 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


template <typename T>
void to_dotprod_test_x<T>::test_ijk_jik(size_t ni, size_t nj, size_t nk)
    throw (libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_dotprod_test_x<T>::test_ijk_jik(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    typedef allocator<T> allocator_t;

    try {

        index<3> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nj - 1; ia2[2] = nk - 1;
        index<3> ib1, ib2;
        ib2[0] = nj - 1; ib2[1] = ni - 1; ib2[2] = nk - 1;
        dimensions<3> dima(index_range<3>(ia1, ia2));
        dimensions<3> dimb(index_range<3>(ib1, ib2));
        size_t sza = dima.get_size(), szb = dimb.get_size();

        dense_tensor<3, T, allocator_t> ta(dima);
        dense_tensor<3, T, allocator_t> tb(dimb);

        T c_ref = 0.0;
        {
            dense_tensor_wr_ctrl<3, T> tca(ta);
            dense_tensor_wr_ctrl<3, T> tcb(tb);
            T *dta = tca.req_dataptr();
            T *dtb = tcb.req_dataptr();

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
        T c1 = to_dotprod<3, T>(ta, pa, tb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        T c2 = to_dotprod<3, T>(ta, pa, tb, pb).calculate();
        pa.permute(1, 2);
        pb.permute(1, 2);
        T c3 = to_dotprod<3, T>(ta, pa, tb, pb).calculate();
        pa.permute(0, 2);
        pb.permute(0, 2);
        T c4 = to_dotprod<3, T>(ta, pa, tb, pb).calculate();

        // Compare against the reference

        if(std::abs(c1 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << c1 << " (result), "
                << c_ref << " (reference), " << c1 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c2 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << c2 << " (result), "
                << c_ref << " (reference), " << c2 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c3 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (3) doesn't match reference: " << c3 << " (result), "
                << c_ref << " (reference), " << c3 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c4 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (4) doesn't match reference: " << c4 << " (result), "
                << c_ref << " (reference), " << c4 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


template <typename T>
void to_dotprod_test_x<T>::test_ijk_jki(size_t ni, size_t nj, size_t nk)
    throw (libtest::test_exception) {

    std::ostringstream tnss;
    tnss << "to_dotprod_test_x<T>::test_ijk_jki(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    typedef allocator<T> allocator_t;

    try {

        index<3> ia1, ia2;
        ia2[0] = ni - 1; ia2[1] = nj - 1; ia2[2] = nk - 1;
        index<3> ib1, ib2;
        ib2[0] = nj - 1; ib2[1] = nk - 1; ib2[2] = ni - 1;
        dimensions<3> dima(index_range<3>(ia1, ia2));
        dimensions<3> dimb(index_range<3>(ib1, ib2));
        size_t sza = dima.get_size(), szb = dimb.get_size();

        dense_tensor<3, T, allocator_t> ta(dima);
        dense_tensor<3, T, allocator_t> tb(dimb);

        T c_ref = 0.0;
        {
            dense_tensor_wr_ctrl<3, T> tca(ta);
            dense_tensor_wr_ctrl<3, T> tcb(tb);
            T *dta = tca.req_dataptr();
            T *dtb = tcb.req_dataptr();

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
        T c1 = to_dotprod<3, T>(ta, pa, tb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        T c2 = to_dotprod<3, T>(ta, pa, tb, pb).calculate();
        pa.permute(1, 2);
        pb.permute(1, 2);
        T c3 = to_dotprod<3, T>(ta, pa, tb, pb).calculate();
        pa.permute(0, 2);
        pb.permute(0, 2);
        T c4 = to_dotprod<3, T>(ta, pa, tb, pb).calculate();

        // Compare against the reference

        if(std::abs(c1 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (1) doesn't match reference: " << c1 << " (result), "
                << c_ref << " (reference), " << c1 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c2 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (2) doesn't match reference: " << c2 << " (result), "
                << c_ref << " (reference), " << c2 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c3 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (3) doesn't match reference: " << c3 << " (result), "
                << c_ref << " (reference), " << c3 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }
        if(std::abs(c4 - c_ref) > k_thresh * std::abs(c_ref)) {
            std::ostringstream ss;
            ss << "Result (4) doesn't match reference: " << c4 << " (result), "
                << c_ref << " (reference), " << c4 - c_ref << " (diff)";
            fail_test(tn.c_str(), __FILE__, __LINE__, ss.str().c_str());
        }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}

template class to_dotprod_test_x<double>;
template class to_dotprod_test_x<float>;

} // namespace libtensor
