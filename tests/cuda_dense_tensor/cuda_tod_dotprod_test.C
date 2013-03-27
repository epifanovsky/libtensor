#include <cmath> // for fabs()
#include <sstream>
#include <libtensor/cuda/cuda_allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_dotprod.h>
#include <libtensor/dense_tensor/tod_random.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_copy_h2d.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_dotprod.h>
#include "cuda_tod_dotprod_test.h"

namespace libtensor {


void cuda_tod_dotprod_test::perform() throw(libtest::test_exception) {

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


void cuda_tod_dotprod_test::test_i_i(size_t ni) {

    std::ostringstream tnss;
    tnss << "cuda_tod_dotprod_test::test_i_i(" << ni << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;
    typedef cuda_allocator<double> cuda_allocator_t;

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
        dense_tensor<1, double, cuda_allocator_t> dta(dima);
        dense_tensor<1, double, cuda_allocator_t> dtb(dimb);

        tod_random<1>().perform(ta);
        tod_random<1>().perform(tb);
        cuda_tod_copy_h2d<1>(ta).perform(dta);
        cuda_tod_copy_h2d<1>(tb).perform(dtb);

        permutation<1> pa, pb;

        double c_ref = tod_dotprod<1>(ta, tb).calculate();

        double c1 = cuda_tod_dotprod<1>(dta, dtb).calculate();
        double c2 = cuda_tod_dotprod<1>(dta, pa, dtb, pb).calculate();

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


void cuda_tod_dotprod_test::test_ij_ij(size_t ni, size_t nj) {

    std::ostringstream tnss;
    tnss << "cuda_tod_dotprod_test::test_ij_ij(" << ni << ", " << nj << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;
    typedef cuda_allocator<double> cuda_allocator_t;

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
        dense_tensor<2, double, cuda_allocator_t> dta(dima);
        dense_tensor<2, double, cuda_allocator_t> dtb(dimb);

        tod_random<2>().perform(ta);
        tod_random<2>().perform(tb);
        cuda_tod_copy_h2d<2>(ta).perform(dta);
        cuda_tod_copy_h2d<2>(tb).perform(dtb);

        permutation<2> pa, pb;

        double c_ref = tod_dotprod<2>(ta, tb).calculate();

        double c1 = cuda_tod_dotprod<2>(dta, dtb).calculate();
        double c2 = cuda_tod_dotprod<2>(dta, pa, dtb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        double c3 = cuda_tod_dotprod<2>(dta, pa, dtb, pb).calculate();

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


void cuda_tod_dotprod_test::test_ij_ji(size_t ni, size_t nj) {

    std::ostringstream tnss;
    tnss << "cuda_tod_dotprod_test::test_ij_ji(" << ni << ", " << nj << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;
    typedef cuda_allocator<double> cuda_allocator_t;

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
        dense_tensor<2, double, cuda_allocator_t> dta(dima);
        dense_tensor<2, double, cuda_allocator_t> dtb(dimb);

        tod_random<2>().perform(ta);
        tod_random<2>().perform(tb);
        cuda_tod_copy_h2d<2>(ta).perform(dta);
        cuda_tod_copy_h2d<2>(tb).perform(dtb);

        permutation<2> pa, pb;
        pb.permute(0, 1);

        double c_ref = tod_dotprod<2>(ta, pa, tb, pb).calculate();

        double c1 = cuda_tod_dotprod<2>(dta, pa, dtb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        double c2 = cuda_tod_dotprod<2>(dta, pa, dtb, pb).calculate();

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


void cuda_tod_dotprod_test::test_ijk_ijk(size_t ni, size_t nj, size_t nk) {

    std::ostringstream tnss;
    tnss << "cuda_tod_dotprod_test::test_ijk_ijk(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;
    typedef cuda_allocator<double> cuda_allocator_t;

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
        dense_tensor<3, double, cuda_allocator_t> dta(dima);
        dense_tensor<3, double, cuda_allocator_t> dtb(dimb);

        tod_random<3>().perform(ta);
        tod_random<3>().perform(tb);
        cuda_tod_copy_h2d<3>(ta).perform(dta);
        cuda_tod_copy_h2d<3>(tb).perform(dtb);

        double c_ref = tod_dotprod<3>(ta, tb).calculate();

        permutation<3> pa, pb;
        double c1 = cuda_tod_dotprod<3>(dta, dtb).calculate();
        double c2 = cuda_tod_dotprod<3>(dta, pa, dtb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        double c3 = cuda_tod_dotprod<3>(dta, pa, dtb, pb).calculate();
        pa.permute(1, 2);
        pb.permute(1, 2);
        double c4 = cuda_tod_dotprod<3>(dta, pa, dtb, pb).calculate();

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


void cuda_tod_dotprod_test::test_ijk_ikj(size_t ni, size_t nj, size_t nk) {

    std::ostringstream tnss;
    tnss << "cuda_tod_dotprod_test::test_ijk_ikj(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;
    typedef cuda_allocator<double> cuda_allocator_t;

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
        dense_tensor<3, double, cuda_allocator_t> dta(dima);
        dense_tensor<3, double, cuda_allocator_t> dtb(dimb);

        tod_random<3>().perform(ta);
        tod_random<3>().perform(tb);
        cuda_tod_copy_h2d<3>(ta).perform(dta);
        cuda_tod_copy_h2d<3>(tb).perform(dtb);

        permutation<3> pa, pb;
        pb.permute(1, 2);

        double c_ref = tod_dotprod<3>(ta, pa, tb, pb).calculate();

        double c1 = cuda_tod_dotprod<3>(dta, pa, dtb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        double c2 = cuda_tod_dotprod<3>(dta, pa, dtb, pb).calculate();
        pa.permute(1, 2);
        pb.permute(1, 2);
        double c3 = cuda_tod_dotprod<3>(dta, pa, dtb, pb).calculate();
        pa.permute(0, 2);
        pb.permute(0, 2);
        double c4 = cuda_tod_dotprod<3>(dta, pa, dtb, pb).calculate();

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


void cuda_tod_dotprod_test::test_ijk_jik(size_t ni, size_t nj, size_t nk) {

    std::ostringstream tnss;
    tnss << "cuda_tod_dotprod_test::test_ijk_jik(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;
    typedef cuda_allocator<double> cuda_allocator_t;

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
        dense_tensor<3, double, cuda_allocator_t> dta(dima);
        dense_tensor<3, double, cuda_allocator_t> dtb(dimb);

        tod_random<3>().perform(ta);
        tod_random<3>().perform(tb);
        cuda_tod_copy_h2d<3>(ta).perform(dta);
        cuda_tod_copy_h2d<3>(tb).perform(dtb);

        permutation<3> pa, pb;
        pb.permute(0, 1);

        double c_ref = tod_dotprod<3>(ta, pa, tb, pb).calculate();

        double c1 = cuda_tod_dotprod<3>(dta, pa, dtb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        double c2 = cuda_tod_dotprod<3>(dta, pa, dtb, pb).calculate();
        pa.permute(1, 2);
        pb.permute(1, 2);
        double c3 = cuda_tod_dotprod<3>(dta, pa, dtb, pb).calculate();
        pa.permute(0, 2);
        pb.permute(0, 2);
        double c4 = cuda_tod_dotprod<3>(dta, pa, dtb, pb).calculate();

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


void cuda_tod_dotprod_test::test_ijk_jki(size_t ni, size_t nj, size_t nk) {

    std::ostringstream tnss;
    tnss << "cuda_tod_dotprod_test::test_ijk_jki(" << ni << ", " << nj << ", "
        << nk << ")";
    std::string tn = tnss.str();

    typedef std_allocator<double> allocator_t;
    typedef cuda_allocator<double> cuda_allocator_t;

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
        dense_tensor<3, double, cuda_allocator_t> dta(dima);
        dense_tensor<3, double, cuda_allocator_t> dtb(dimb);

        tod_random<3>().perform(ta);
        tod_random<3>().perform(tb);
        cuda_tod_copy_h2d<3>(ta).perform(dta);
        cuda_tod_copy_h2d<3>(tb).perform(dtb);

        permutation<3> pa, pb;
        pb.permute(1, 2).permute(0, 1); // jki -> ijk

        double c_ref = tod_dotprod<3>(ta, pa, tb, pb).calculate();

        double c1 = cuda_tod_dotprod<3>(dta, pa, dtb, pb).calculate();
        pa.permute(0, 1);
        pb.permute(0, 1);
        double c2 = cuda_tod_dotprod<3>(dta, pa, dtb, pb).calculate();
        pa.permute(1, 2);
        pb.permute(1, 2);
        double c3 = cuda_tod_dotprod<3>(dta, pa, dtb, pb).calculate();
        pa.permute(0, 2);
        pb.permute(0, 2);
        double c4 = cuda_tod_dotprod<3>(dta, pa, dtb, pb).calculate();

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
