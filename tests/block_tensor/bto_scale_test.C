#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/bto_scale.h>
#include <libtensor/block_tensor/bto_random.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/dense_tensor/to_btconv.h>
#include "bto_scale_test.h"
#include "../compare_ref.h"

namespace libtensor {


void bto_scale_test::perform() throw(libtest::test_exception) {
    std::cout << "Testing bto_scale_test_x<double>   ";
    bto_scale_test_x<double> t_double;
    t_double.perform();
    std::cout << "Testing bto_scale_test_x<float>   ";
    bto_scale_test_x<float> t_float;
    t_float.perform();
}

template<>
const double bto_scale_test_x<double>::k_thresh = 7e-14;

template<>
const float bto_scale_test_x<float>::k_thresh = 2e-5;


template<typename T>
void bto_scale_test_x<T>::perform() throw(libtest::test_exception) {

    allocator<T>::init(4, 16, 65536, 65536);
    try {

    test_0();
    test_i(3);
    test_i(10);
    test_i(32);

    test_1();

    } catch (...) {
        allocator<T>::shutdown();
        throw;
    }
    allocator<T>::shutdown();
}


template<typename T>
template<size_t N>
void bto_scale_test_x<T>::test_generic(
    const char *testname, block_tensor_i<N, T> &bt, T c)
    throw(libtest::test_exception) {

    try {

    dense_tensor<N, T, allocator<T> > t(bt.get_bis().get_dims()),
        t_ref(bt.get_bis().get_dims());
    to_btconv<N, T>(bt).perform(t_ref);
    to_scale<N, T>(c).perform(t_ref);

    bto_scale<N, T>(bt, c).perform();
    to_btconv<N, T>(bt).perform(t);

    compare_ref_x<N, T>::compare(testname, t, t_ref, 1e-15);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Checks that scaling by zero results in all zero blocks
 **/
template<typename T>
void bto_scale_test_x<T>::test_0() throw(libtest::test_exception) {

    static const char *testname = "bto_scale_test_x<T>::test_0()";

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m;
    m[0] = true; m[1] = true; m[2] = true; m[3] = true;
    bis.split(m, 3); bis.split(m, 7);
    dimensions<4> bidims(bis.get_block_index_dims());

    volatile T zero = 0.0;
    volatile T n_one = -1.0;
    volatile T n_zero = n_one * zero;

    block_tensor<4, T, allocator<T> > bt1(bis), bt2(bis);
    bto_random<4, T>().perform(bt1);
    bto_random<4, T>().perform(bt2);
    test_generic(testname, bt1, zero);
    test_generic(testname, bt2, n_zero);

    block_tensor_ctrl<4, T> ctrl1(bt1);
    abs_index<4> ai1(bidims);
    do {
        if(!ctrl1.req_is_zero_block(ai1.get_index())) {
            fail_test(testname, __FILE__, __LINE__,
                "Bad zero block structure in bt1.");
        }
    } while(ai1.inc());

    block_tensor_ctrl<4, T> ctrl2(bt2);
    abs_index<4> ai2(bidims);
    do {
        if(!ctrl2.req_is_zero_block(ai2.get_index())) {
            fail_test(testname, __FILE__, __LINE__,
                "Bad zero block structure in bt2.");
        }
    } while(ai2.inc());

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


/** \test Checks the scaling and the zero-block structure of one-dim
        tensors
 **/
template<typename T>
void bto_scale_test_x<T>::test_i(size_t i) throw(libtest::test_exception) {

    std::ostringstream ss;
    ss << "bto_scale_test_x<T>::test_i(" << i << ")";
    std::string tn = ss.str();

    try {

    index<1> ia, ib;
    ia[0] = i - 1;
    dimensions<1> d(index_range<1>(ia, ib));
    mask<1> m; m[0] = true;

    block_index_space<1> bis1(d);
    block_index_space<1> bis2(d);
    bis2.split(m, i/2);
    block_index_space<1> bis3(d);
    bis3.split(m, i/3); bis3.split(m, 2*i/3);

    //  Test the correct scaling

    block_tensor<1, T, allocator<T> > bt1a(bis1);
    block_tensor<1, T, allocator<T> > bt2a(bis2);
    block_tensor<1, T, allocator<T> > bt3a(bis3);
    bto_random<1, T>().perform(bt1a);
    bto_random<1, T>().perform(bt2a);
    bto_random<1, T>().perform(bt3a);
    test_generic(tn.c_str(), bt1a, 0.5);
    test_generic(tn.c_str(), bt2a, -1.5);
    test_generic(tn.c_str(), bt3a, 2.2);

    index<1> i0, i1, i2;
    i1[0] = 1; i2[0] = 2;

    //  Test the correct zero block structure

    block_tensor<1, T, allocator<T> > bt1b(bis1);
    block_tensor<1, T, allocator<T> > bt2b(bis2);
    block_tensor<1, T, allocator<T> > bt3b(bis3);
    bto_random<1, T>().perform(bt2b, i1);
    bto_random<1, T>().perform(bt3b, i0);
    bto_random<1, T>().perform(bt3b, i2);
    test_generic(tn.c_str(), bt1b, 1.0);
    test_generic(tn.c_str(), bt2b, -0.6);
    test_generic(tn.c_str(), bt3b, -2.7);

    block_tensor_ctrl<1, T> c1b(bt1b), c2b(bt2b), c3b(bt3b);
    if(!c1b.req_is_zero_block(i0)) {
        fail_test(tn.c_str(), __FILE__, __LINE__,
            "!c1b.req_is_zero_block(i0)");
    }
    if(!c2b.req_is_zero_block(i0)) {
        fail_test(tn.c_str(), __FILE__, __LINE__,
            "!c2b.req_is_zero_block(i0)");
    }
    if(c2b.req_is_zero_block(i1)) {
        fail_test(tn.c_str(), __FILE__, __LINE__,
            "c2b.req_is_zero_block(i1)");
    }
    if(c3b.req_is_zero_block(i0)) {
        fail_test(tn.c_str(), __FILE__, __LINE__,
            "c3b.req_is_zero_block(i0)");
    }
    if(!c3b.req_is_zero_block(i1)) {
        fail_test(tn.c_str(), __FILE__, __LINE__,
            "!c3b.req_is_zero_block(i1)");
    }
    if(c3b.req_is_zero_block(i2)) {
        fail_test(tn.c_str(), __FILE__, __LINE__,
            "c3b.req_is_zero_block(i2)");
    }

    } catch(exception &e) {
        fail_test(tn.c_str(), __FILE__, __LINE__, e.what());
    }
}


/** \test Tests proper scaling in block tensors with permutational
        %symmetry and anti-symmetry.
 **/
template<typename T>
void bto_scale_test_x<T>::test_1() throw(libtest::test_exception) {

    static const char *testname = "bto_scale_test_x<T>::test_1()";

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m;
    m[0] = true; m[1] = true; m[2] = true; m[3] = true;
    bis.split(m, 3); bis.split(m, 7);

    block_tensor<4, T, allocator<T> > bt1(bis), bt2(bis);

    {
        block_tensor_ctrl<4, T> ctrl1(bt1), ctrl2(bt2);
        scalar_transf<T> tr0, tr1(-1.);
        se_perm<4, T> elem1(permutation<4>().permute(1, 2), tr0);
        se_perm<4, T> elem2(permutation<4>().permute(0, 1).
            permute(1, 2).permute(2, 3), tr0);
        se_perm<4, T> elem3(permutation<4>().permute(0, 1), tr1);
        se_perm<4, T> elem4(permutation<4>().permute(2, 3), tr1);
        ctrl1.req_symmetry().insert(elem1);
        ctrl1.req_symmetry().insert(elem2);
        ctrl2.req_symmetry().insert(elem3);
        ctrl2.req_symmetry().insert(elem4);
    }

    bto_random<4, T>().perform(bt1);
    bto_random<4, T>().perform(bt2);

    test_generic(testname, bt1, 0.45);
    test_generic(testname, bt2, -1.0);

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
