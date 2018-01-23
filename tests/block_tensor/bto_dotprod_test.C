#include <cmath>
#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/bto_dotprod.h>
#include <libtensor/block_tensor/bto_random.h>
#include <libtensor/symmetry/point_group_table.h>
#include <libtensor/symmetry/product_table_container.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/se_perm.h>
#include <libtensor/dense_tensor/to_btconv.h>
#include <libtensor/dense_tensor/to_dotprod.h>
#include "bto_dotprod_test.h"

namespace libtensor {

void bto_dotprod_test::perform() throw(libtest::test_exception) {
    std::cout << "Testing bto_dotprod_test_x<double>  ";
    bto_dotprod_test_x<double> t_double;
    t_double.perform();
    std::cout << "Testing bto_dotprod_test_x<float>  ";
    bto_dotprod_test_x<float> t_float;
    t_float.perform();
}

template<>
const double bto_dotprod_test_x<double>::k_thresh = 1e-14;

template<>
const float bto_dotprod_test_x<float>::k_thresh = 7e-6;

template<typename T>
void bto_dotprod_test_x<T>::perform() throw(libtest::test_exception) {

    allocator<T>::init(4, 16, 65536, 65536);
    try {

    test_1();
    test_2();
    test_3();
    test_4();
    test_5();
    test_6();
    test_7();
    test_8();
    test_9();
    test_10a();
    test_10b();
    test_10c(true);
    test_10c(false);
    test_11();
    test_12();
    test_13a();
    test_13b();

    }
    catch (...) {
        allocator<T>::shutdown();
        throw;
    }
    allocator<T>::shutdown();
}


template<typename T>
void bto_dotprod_test_x<T>::test_1() throw(libtest::test_exception) {

    //
    //  Single block, both arguments are non-zero
    //

    static const char *testname = "bto_dotprod_test_x<T>::test_1()";

    typedef allocator<T> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 8; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    block_tensor<2, T, allocator_t> bt1(bis), bt2(bis);

    //  Fill in random data

    bto_random<2, T>().perform(bt1);
    bto_random<2, T>().perform(bt2);

    //  Compute the dot product

    T d = bto_dotprod<2, T>(bt1, bt2).calculate();

    //  Compute the reference

    dense_tensor<2, T, allocator_t> t1(dims), t2(dims);
    to_btconv<2, T>(bt1).perform(t1);
    to_btconv<2, T>(bt2).perform(t2);
    T d_ref = to_dotprod<2, T>(t1, t2).calculate();

    //  Compare

    if(std::abs(d - d_ref) > 10*k_thresh) {
        std::ostringstream ss;
        ss << "Result does not match reference: " << d << " vs. "
            << d_ref << " (ref), " << d - d_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void bto_dotprod_test_x<T>::test_2() throw(libtest::test_exception) {

    //
    //  Single block, one of the arguments is zero
    //

    static const char *testname = "bto_dotprod_test_x<T>::test_2()";

    typedef allocator<T> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 8; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    block_tensor<2, T, allocator_t> bt1(bis), bt2(bis);

    //  Fill in random data

    bto_random<2, T>().perform(bt1);

    //  Compute the dot product

    T d = bto_dotprod<2, T>(bt1, bt2).calculate();

    //  Compare

    T d_ref = 0.0;
    if(std::abs(d) != 0.0) {
        std::ostringstream ss;
        ss << "Result does not match reference: " << d << " vs. "
            << d_ref << " (ref), " << d - d_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void bto_dotprod_test_x<T>::test_3() throw(libtest::test_exception) {

    //
    //  Two blocks in each dimension, both arguments are non-zero
    //

    static const char *testname = "bto_dotprod_test_x<T>::test_3()";

    typedef allocator<T> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 8; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m1, m2;
    m1[0] = true; m2[1] = true;
    bis.split(m1, 5);
    bis.split(m2, 2);
    block_tensor<2, T, allocator_t> bt1(bis), bt2(bis);

    //  Fill in random data

    bto_random<2, T>().perform(bt1);
    bto_random<2, T>().perform(bt2);

    //  Compute the dot product

    T d = bto_dotprod<2, T>(bt1, bt2).calculate();

    //  Compute the reference

    dense_tensor<2, T, allocator_t> t1(dims), t2(dims);
    to_btconv<2, T>(bt1).perform(t1);
    to_btconv<2, T>(bt2).perform(t2);
    T d_ref = to_dotprod<2, T>(t1, t2).calculate();

    //  Compare

    if(std::abs(d - d_ref) > 10*k_thresh) {
        std::ostringstream ss;
        ss << "Result does not match reference: " << d << " vs. "
            << d_ref << " (ref), " << d - d_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void bto_dotprod_test_x<T>::test_4() throw(libtest::test_exception) {

    //
    //  Two blocks in each dimension, off-diagonal blocks of one of
    //  the arguments are zero
    //

    static const char *testname = "bto_dotprod_test_x<T>::test_4()";

    typedef allocator<T> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 8; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m1, m2;
    m1[0] = true; m2[1] = true;
    bis.split(m1, 5);
    bis.split(m2, 2);
    block_tensor<2, T, allocator_t> bt1(bis), bt2(bis);

    //  Fill in random data

    bto_random<2, T>().perform(bt1);
    i1[0] = 0; i1[1] = 0;
    i2[0] = 1; i2[1] = 1;
    bto_random<2, T>().perform(bt2, i1);
    bto_random<2, T>().perform(bt2, i2);

    //  Compute the dot product

    T d = bto_dotprod<2, T>(bt1, bt2).calculate();

    //  Compute the reference

    dense_tensor<2, T, allocator_t> t1(dims), t2(dims);
    to_btconv<2, T>(bt1).perform(t1);
    to_btconv<2, T>(bt2).perform(t2);
    T d_ref = to_dotprod<2, T>(t1, t2).calculate();

    //  Compare

    if(std::abs(d - d_ref) > 10*k_thresh) {
        std::ostringstream ss;
        ss << "Result does not match reference: " << d << " vs. "
            << d_ref << " (ref), " << d - d_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void bto_dotprod_test_x<T>::test_5() throw(libtest::test_exception) {

    //
    //  Two blocks in each dimension, multiple non-zero arguments
    //

    static const char *testname = "bto_dotprod_test_x<T>::test_5()";

    typedef allocator<T> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 8; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m1, m2;
    m1[0] = true; m2[1] = true;
    bis.split(m1, 5);
    bis.split(m2, 2);
    block_tensor<2, T, allocator_t> bt1(bis), bt2(bis), bt3(bis),
        bt4(bis);

    //  Fill in random data

    bto_random<2, T>().perform(bt1);
    bto_random<2, T>().perform(bt2);
    bto_random<2, T>().perform(bt3);
    bto_random<2, T>().perform(bt4);
    bt1.set_immutable();
    bt2.set_immutable();
    bt3.set_immutable();
    bt4.set_immutable();

    //  Compute the dot product

    bto_dotprod<2, T> op(bt1, bt2);
    op.add_arg(bt3, bt4);
    std::vector<T> v(2);
    op.calculate(v);
    T d1 = v[0], d2 = v[1];

    //  Compute the reference

    dense_tensor<2, T, allocator_t> t1(dims), t2(dims), t3(dims), t4(dims);
    to_btconv<2, T>(bt1).perform(t1);
    to_btconv<2, T>(bt2).perform(t2);
    to_btconv<2, T>(bt3).perform(t3);
    to_btconv<2, T>(bt4).perform(t4);
    T d1_ref = to_dotprod<2, T>(t1, t2).calculate();
    T d2_ref = to_dotprod<2, T>(t3, t4).calculate();

    //  Compare

    if(std::abs(d1 - d1_ref) > 10*k_thresh) {
        std::ostringstream ss;
        ss << "Result 1 does not match reference: " << d1 << " vs. "
            << d1_ref << " (ref), " << d1 - d1_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(std::abs(d2 - d2_ref) > 10*k_thresh) {
        std::ostringstream ss;
        ss << "Result 2 does not match reference: " << d2 << " vs. "
            << d2_ref << " (ref), " << d2 - d2_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void bto_dotprod_test_x<T>::test_6() throw(libtest::test_exception) {

    //
    //  Two blocks in each dimension, multiple non-zero arguments
    //

    static const char *testname = "bto_dotprod_test_x<T>::test_6()";

    typedef allocator<T> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 8; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m1, m2;
    m1[0] = true; m2[1] = true;
    bis.split(m1, 5);
    bis.split(m2, 2);
    block_tensor<2, T, allocator_t> bt1(bis), bt2(bis), bt3(bis),
        bt4(bis);

    //  Fill in random data

    bto_random<2, T>().perform(bt1);
    bto_random<2, T>().perform(bt2);
    bto_random<2, T>().perform(bt3);
    bto_random<2, T>().perform(bt4);
    bt1.set_immutable();
    bt2.set_immutable();
    bt3.set_immutable();
    bt4.set_immutable();

    //  Compute the dot product

    bto_dotprod<2, T> op(bt1, bt2);
    op.add_arg(bt3, bt1);
    op.add_arg(bt1, bt4);
    std::vector<T> v(3);
    op.calculate(v);
    T d1 = v[0], d2 = v[1], d3 = v[2];

    //  Compute the reference

    dense_tensor<2, T, allocator_t> t1(dims), t2(dims), t3(dims), t4(dims);
    to_btconv<2, T>(bt1).perform(t1);
    to_btconv<2, T>(bt2).perform(t2);
    to_btconv<2, T>(bt3).perform(t3);
    to_btconv<2, T>(bt4).perform(t4);
    T d1_ref = to_dotprod<2, T>(t1, t2).calculate();
    T d2_ref = to_dotprod<2, T>(t1, t3).calculate();
    T d3_ref = to_dotprod<2, T>(t1, t4).calculate();

    //  Compare

    if(std::abs(d1 - d1_ref) > 10*k_thresh) {
        std::ostringstream ss;
        ss << "Result 1 does not match reference: " << d1 << " vs. "
            << d1_ref << " (ref), " << d1 - d1_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(std::abs(d2 - d2_ref) > 10*k_thresh) {
        std::ostringstream ss;
        ss << "Result 2 does not match reference: " << d2 << " vs. "
            << d2_ref << " (ref), " << d2 - d2_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(std::abs(d3 - d3_ref) > 10*k_thresh) {
        std::ostringstream ss;
        ss << "Result 3 does not match reference: " << d3 << " vs. "
            << d3_ref << " (ref), " << d3 - d3_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void bto_dotprod_test_x<T>::test_7() throw(libtest::test_exception) {

    //
    //  Three blocks in each dimension, both arguments are non-zero,
    //  permutational symmetry in both arguments
    //

    static const char *testname = "bto_dotprod_test_x<T>::test_7()";

    typedef allocator<T> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m;
    m[0] = true; m[1] = true;
    bis.split(m, 2);
    bis.split(m, 5);
    bis.split(m, 7);
    block_tensor<2, T, allocator_t> bt1(bis), bt2(bis);

    //  Install permutational symmetry

    {
        block_tensor_ctrl<2, T> ctrl1(bt1), ctrl2(bt2);
        scalar_transf<T> tr0, tr1(-1.);
        se_perm<2, T> elem(permutation<2>().permute(0, 1), tr0);
        ctrl1.req_symmetry().insert(elem);
        ctrl2.req_symmetry().insert(elem);
    }

    //  Fill in random data

    bto_random<2, T>().perform(bt1);
    bto_random<2, T>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    //  Compute the dot product

    T d = bto_dotprod<2, T>(bt1, bt2).calculate();

    //  Compute the reference

    dense_tensor<2, T, allocator_t> t1(dims), t2(dims);
    to_btconv<2, T>(bt1).perform(t1);
    to_btconv<2, T>(bt2).perform(t2);
    T d_ref = to_dotprod<2, T>(t1, t2).calculate();

    //  Compare

    if(std::abs(d - d_ref) > 10*k_thresh) {
        std::ostringstream ss;
        ss << "Result does not match reference: " << d << " vs. "
            << d_ref << " (ref), " << d - d_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void bto_dotprod_test_x<T>::test_8() throw(libtest::test_exception) {

    //
    //  Three blocks in each dimension, multiple non-zero arguments,
    //  various kinds of permutational symmetry
    //

    static const char *testname = "bto_dotprod_test_x<T>::test_8()";

    typedef allocator<T> allocator_t;

    try {

    index<4> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9; i2[3] = 9;
    dimensions<4> dims(index_range<4>(i1, i2));
    block_index_space<4> bis(dims);
    mask<4> m;
    m[0] = true; m[1] = true; m[2] = true; m[3] = true;
    bis.split(m, 2);
    bis.split(m, 5);
    bis.split(m, 7);

    block_tensor<4, T, allocator_t> bt1(bis), bt2(bis), bt3(bis),
        bt4(bis), bt5(bis), bt6(bis);

    //  Set up symmetry

    {
        block_tensor_ctrl<4, T> ctrl1(bt1), ctrl2(bt2), ctrl3(bt3),
            ctrl4(bt4), ctrl5(bt5), ctrl6(bt6);

        scalar_transf<T> tr0, tr1(-1.);
        se_perm<4, T> elem1(permutation<4>().permute(0, 1).
            permute(2, 3), tr0);
        se_perm<4, T> elem2(permutation<4>().permute(0, 1).
            permute(2, 3), tr1);
        se_perm<4, T> elem3(permutation<4>().permute(0, 1), tr0);
        se_perm<4, T> elem4(permutation<4>().permute(0, 1).
            permute(1, 2).permute(2, 3), tr0);

        ctrl1.req_symmetry().insert(elem1);
        ctrl2.req_symmetry().insert(elem3);
        ctrl2.req_symmetry().insert(elem4);
        ctrl3.req_symmetry().insert(elem1);
        ctrl4.req_symmetry().insert(elem2);
        ctrl5.req_symmetry().insert(elem1);
        ctrl6.req_symmetry().insert(elem3);
    }

    //  Fill in random data

    bto_random<4, T>().perform(bt1);
    bto_random<4, T>().perform(bt2);
    bto_random<4, T>().perform(bt3);
    bto_random<4, T>().perform(bt4);
    bto_random<4, T>().perform(bt5);
    bto_random<4, T>().perform(bt6);
    bt1.set_immutable();
    bt2.set_immutable();
    bt3.set_immutable();
    bt4.set_immutable();
    bt5.set_immutable();
    bt6.set_immutable();

    //  Compute the dot product

    bto_dotprod<4, T> op(bt1, permutation<4>(), bt2,
        permutation<4>().permute(1, 2));
    op.add_arg(bt3, bt4);
    op.add_arg(bt5, permutation<4>().permute(0, 2).permute(1, 3), bt6,
        permutation<4>());
    std::vector<T> v(3);
    op.calculate(v);
    T d1 = v[0], d2 = v[1], d3 = v[2];

    //  Compute the reference

    dense_tensor<4, T, allocator_t> t1(dims), t2(dims), t3(dims), t4(dims),
        t5(dims), t6(dims);
    to_btconv<4, T>(bt1).perform(t1);
    to_btconv<4, T>(bt2).perform(t2);
    to_btconv<4, T>(bt3).perform(t3);
    to_btconv<4, T>(bt4).perform(t4);
    to_btconv<4, T>(bt5).perform(t5);
    to_btconv<4, T>(bt6).perform(t6);
    T d1_ref = to_dotprod<4, T>(t1, t2).calculate();
    T d2_ref = to_dotprod<4, T>(t3, t4).calculate();
    T d3_ref = to_dotprod<4, T>(t5, permutation<4>().permute(0, 2).
        permute(1, 3), t6, permutation<4>()).calculate();

    //  Compare

    if(std::abs(d1 - d1_ref) > std::abs(d1_ref) * k_thresh) {
        std::ostringstream ss;
        ss << "Result 1 does not match reference: " << d1 << " vs. "
            << d1_ref << " (ref), " << d1 - d1_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(std::abs(d2 - d2_ref) > 1000*k_thresh) {
        std::ostringstream ss;
        ss << "Result 2 does not match reference: " << d2 << " vs. "
            << d2_ref << " (ref), " << d2 - d2_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }
    if(std::abs(d3 - d3_ref) > std::abs(d3_ref) * k_thresh) {
        std::ostringstream ss;
        ss << "Result 3 does not match reference: " << d3 << " vs. "
            << d3_ref << " (ref), " << d3 - d3_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

template<typename T>
void bto_dotprod_test_x<T>::test_9() throw(libtest::test_exception) {

    //
    //  Four blocks in each dimension, multiple non-zero arguments,
    //  permutational anti-symmetry
    //

    static const char *testname = "bto_dotprod_test_x<T>::test_9()";

    typedef allocator<T> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m;
    m[0] = true; m[1] = true;
    bis.split(m, 2);
    bis.split(m, 5);
    bis.split(m, 7);
    block_tensor<2, T, allocator_t> bt1(bis), bt2(bis);

    //  Set up symmetry

    {
        block_tensor_ctrl<2, T> ctrl1(bt1), ctrl2(bt2);

        scalar_transf<T> tr0, tr1(-1.);
        se_perm<2, T> elem1(permutation<2>().permute(0, 1), tr1);

        ctrl1.req_symmetry().insert(elem1);
        ctrl2.req_symmetry().insert(elem1);
    }

    //  Fill in random data

    bto_random<2, T>().perform(bt1);
    bto_random<2, T>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    //  Compute the dot product

    T d = bto_dotprod<2, T>(bt1, bt2).calculate();

    //  Compute the reference

    dense_tensor<2, T, allocator_t> t1(dims), t2(dims);
    to_btconv<2, T>(bt1).perform(t1);
    to_btconv<2, T>(bt2).perform(t2);
    T d_ref = to_dotprod<2, T>(t1, t2).calculate();

    //  Compare

    if(std::abs(d - d_ref) > std::abs(d_ref) * k_thresh) {
        std::ostringstream ss;
        ss << "Result does not match reference: " << d << " vs. "
            << d_ref << " (ref), " << d - d_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

template<typename T>
void bto_dotprod_test_x<T>::test_10a() throw(libtest::test_exception) {

    //
    //  Four blocks in each dimension, multiple non-zero arguments,
    //  label symmetry, but only in one block tensor
    //

    std::ostringstream tnss;
    tnss << "bto_dotprod_test_x<T>::test_10a()";

    typedef allocator<T> allocator_t;

    //
    // Setup product table
    //
    {
        std::vector<std::string> irnames(2);
        irnames[0] = "g"; irnames[1] = "u";
        point_group_table pg(tnss.str(), irnames, irnames[0]);
        pg.add_product(1, 1, 0);

        product_table_container::get_instance().add(pg);
    }

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m;
    m[0] = true; m[1] = true;
    bis.split(m, 2);
    bis.split(m, 5);
    bis.split(m, 7);
    block_tensor<2, T, allocator_t> bt1(bis), bt2(bis);

    //  Set up symmetry

    {
        block_tensor_ctrl<2, T> ctrl1(bt1);

        se_label<2, T> l(bis.get_block_index_dims(), tnss.str());
        block_labeling<2> &bl = l.get_labeling();
        bl.assign(m, 0, 0);
        bl.assign(m, 1, 1);
        bl.assign(m, 2, 0);
        bl.assign(m, 3, 1);
        l.set_rule(0);

        ctrl1.req_symmetry().insert(l);
    }

    //  Fill in random data

    bto_random<2, T>().perform(bt1);
    bto_random<2, T>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    //  Compute the dot product

    T d = bto_dotprod<2, T>(bt1, bt2).calculate();

    libutil::timings_store<libtensor_timings>::get_instance().print(std::cout);

    //  Compute the reference

    dense_tensor<2, T, allocator_t> t1(dims), t2(dims);
    to_btconv<2, T>(bt1).perform(t1);
    to_btconv<2, T>(bt2).perform(t2);
    T d_ref = to_dotprod<2, T>(t1, t2).calculate();

    //  Compare

    if(std::abs(d - d_ref) > std::abs(d_ref) * k_thresh) {
        std::ostringstream ss;
        ss << "Result does not match reference: " << d << " vs. "
            << d_ref << " (ref), " << d - d_ref << " (diff).";
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        product_table_container::get_instance().erase(tnss.str());
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }

    product_table_container::get_instance().erase(tnss.str());
}


template<typename T>
void bto_dotprod_test_x<T>::test_10b() throw(libtest::test_exception) {

    //
    //  Four blocks in each dimension, multiple non-zero arguments,
    //  label symmetry in both block tensors
    //

    std::ostringstream tnss;
    tnss << "bto_dotprod_test_x<T>::test_10b()";

    typedef allocator<T> allocator_t;

    //
    // Setup product table
    //
    {
        std::vector<std::string> irnames(4);
        irnames[0] = "A1"; irnames[1] = "A2";
        irnames[1] = "B1"; irnames[1] = "B2";
        point_group_table pg(tnss.str(), irnames, irnames[0]);
        pg.add_product(1, 1, 0);
        pg.add_product(1, 2, 3);
        pg.add_product(1, 3, 2);
        pg.add_product(2, 2, 0);
        pg.add_product(2, 3, 1);
        pg.add_product(3, 3, 0);

        product_table_container::get_instance().add(pg);
    }

    try {

    index<2> i1, i2;
    i2[0] = 11; i2[1] = 11;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m;
    m[0] = true; m[1] = true;
    bis.split(m, 2);
    bis.split(m, 5);
    bis.split(m, 6);
    bis.split(m, 8);
    bis.split(m, 11);
    block_tensor<2, T, allocator_t> bt1(bis), bt2(bis);

    //  Set up symmetry

    {
        block_tensor_ctrl<2, T> ctrl1(bt1), ctrl2(bt2);

        se_label<2, T> l(bis.get_block_index_dims(), tnss.str());
        block_labeling<2> &bl = l.get_labeling();
        bl.assign(m, 0, 0);
        bl.assign(m, 1, 2);
        bl.assign(m, 2, 3);
        bl.assign(m, 3, 0);
        bl.assign(m, 4, 2);
        bl.assign(m, 5, 3);
        l.set_rule(2);

        ctrl1.req_symmetry().insert(l);
        l.set_rule(1);
        ctrl2.req_symmetry().insert(l);
    }

    //  Fill in random data

    bto_random<2, T>().perform(bt1);
    bto_random<2, T>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    //  Compute the dot product

    T d = bto_dotprod<2, T>(bt1, bt2).calculate();

    libutil::timings_store<libtensor_timings>::get_instance().print(std::cout);

    //  Compute the reference

    dense_tensor<2, T, allocator_t> t1(dims), t2(dims);
    to_btconv<2, T>(bt1).perform(t1);
    to_btconv<2, T>(bt2).perform(t2);
    T d_ref = to_dotprod<2, T>(t1, t2).calculate();

    //  Compare

    if(std::abs(d - d_ref) > std::abs(d_ref) * k_thresh) {
        std::ostringstream ss;
        ss << "Result does not match reference: " << d << " vs. "
            << d_ref << " (ref), " << d - d_ref << " (diff).";
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        product_table_container::get_instance().erase(tnss.str());
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }

    product_table_container::get_instance().erase(tnss.str());
}


template<typename T>
void bto_dotprod_test_x<T>::test_10c(
        bool both) throw(libtest::test_exception) {

    //
    //  Two blocks in each dimension, multiple non-zero arguments,
    //  partition symmetry in one or both block tensors
    //

    std::ostringstream tnss;
    tnss << "bto_dotprod_test_x<T>::test_10c(" << both << ")";

    typedef allocator<T> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 11; i2[1] = 11;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m;
    m[0] = true; m[1] = true;
    bis.split(m, 2);
    bis.split(m, 6);
    bis.split(m, 8);
    block_tensor<2, T, allocator_t> bt1(bis), bt2(bis);

    //  Set up symmetry

    {
        block_tensor_ctrl<2, T> ctrl1(bt1), ctrl2(bt2);

        scalar_transf<T> tr(1.0);
        se_part<2, T> sp(bis, m, 2);

        index<2> i00, i01, i10, i11;
        i10[0] = i01[1] = i11[0] = i11[1] = 1;
        sp.add_map(i00, i11, tr);
        sp.mark_forbidden(i01);
        sp.mark_forbidden(i10);

        ctrl1.req_symmetry().insert(sp);
        if (both) ctrl2.req_symmetry().insert(sp);
    }

    //  Fill in random data

    bto_random<2, T>().perform(bt1);
    bto_random<2, T>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    //  Compute the dot product

    T d = bto_dotprod<2, T>(bt1, bt2).calculate();

    libutil::timings_store<libtensor_timings>::get_instance().print(std::cout);

    //  Compute the reference

    dense_tensor<2, T, allocator_t> t1(dims), t2(dims);
    to_btconv<2, T>(bt1).perform(t1);
    to_btconv<2, T>(bt2).perform(t2);
    T d_ref = to_dotprod<2, T>(t1, t2).calculate();

    //  Compare

    if(std::abs(d - d_ref) > std::abs(d_ref) * k_thresh) {
        std::ostringstream ss;
        ss << "Result does not match reference: " << d << " vs. "
            << d_ref << " (ref), " << d - d_ref << " (diff).";
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void bto_dotprod_test_x<T>::test_11() throw(libtest::test_exception) {

    //
    //  Two blocks in each dimension, subtle splitting differences
    //  producing same, but not identical block index spaces
    //

    static const char *testname = "bto_dotprod_test_x<T>::test_11()";

    typedef allocator<T> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis1(dims), bis2(dims);
    mask<2> m01, m10, m11;
    m10[0] = true; m01[1] = true;
    m11[0] = true; m11[1] = true;
    bis1.split(m01, 5);
    bis1.split(m10, 5);
    bis2.split(m11, 5);
    block_tensor<2, T, allocator_t> bt1(bis1), bt2(bis2);

    //  Fill in random data

    bto_random<2, T>().perform(bt1);
    bto_random<2, T>().perform(bt2);

    //  Compute the dot product

    T d = bto_dotprod<2, T>(bt1, bt2).calculate();

    //  Compute the reference

    dense_tensor<2, T, allocator_t> t1(dims), t2(dims);
    to_btconv<2, T>(bt1).perform(t1);
    to_btconv<2, T>(bt2).perform(t2);
    T d_ref = to_dotprod<2, T>(t1, t2).calculate();

    //  Compare

    if(std::abs(d - d_ref) > 10*k_thresh) {
        std::ostringstream ss;
        ss << "Result does not match reference: " << d << " vs. "
            << d_ref << " (ref), " << d - d_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void bto_dotprod_test_x<T>::test_12() throw(libtest::test_exception) {

    //
    //  Two blocks in each dimension, subtle splitting differences
    //  producing same, but not identical block index spaces
    //

    static const char *testname = "bto_dotprod_test_x<T>::test_12()";

    typedef allocator<T> allocator_t;

    try {

    index<2> i1, i2;
    i2[0] = 9; i2[1] = 9;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis1(dims), bis2(dims);
    mask<2> m01, m10, m11;
    m10[0] = true; m01[1] = true;
    m11[0] = true; m11[1] = true;
    bis1.split(m01, 5);
    bis1.split(m10, 5);
    bis2.split(m11, 5);
    block_tensor<2, T, allocator_t> bt1(bis1), bt2(bis2);

    //  Fill in random data

    bto_random<2, T>().perform(bt1);
    bto_random<2, T>().perform(bt2);

    //  Compute the dot product

    permutation<2> p01, p10;
    p10.permute(0, 1);
    T d = bto_dotprod<2, T>(bt1, p10, bt2, p01).calculate();

    //  Compute the reference

    dense_tensor<2, T, allocator_t> t1(dims), t2(dims);
    to_btconv<2, T>(bt1).perform(t1);
    to_btconv<2, T>(bt2).perform(t2);
    T d_ref = to_dotprod<2, T>(t1, p10, t2, p01).calculate();

    //  Compare

    if(std::abs(d - d_ref) > 10*k_thresh) {
        std::ostringstream ss;
        ss << "Result does not match reference: " << d << " vs. "
            << d_ref << " (ref), " << d - d_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void bto_dotprod_test_x<T>::test_13a() throw(libtest::test_exception) {

    //  3-dim tensor, four blocks in each dimension,
    //  permutational anti-symmetry

    static const char testname[] = "bto_dotprod_test_x<T>::test_13a()";

    typedef allocator<T> allocator_t;

    try {

    index<3> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9;
    dimensions<3> dims(index_range<3>(i1, i2));
    block_index_space<3> bis(dims);
    mask<3> m;
    m[0] = true; m[1] = true; m[2] = true;
    bis.split(m, 2);
    bis.split(m, 5);
    bis.split(m, 7);
    block_tensor<3, T, allocator_t> bt1(bis), bt2(bis);

    //  Set up symmetry

    {
        block_tensor_ctrl<3, T> ctrl1(bt1), ctrl2(bt2);

        scalar_transf<T> tr0, tr1(-1.);
        se_perm<3, T> elem1(permutation<3>().permute(0, 1), tr1);
        se_perm<3, T> elem2(permutation<3>().permute(1, 2), tr1);

        ctrl1.req_symmetry().insert(elem1);
        ctrl1.req_symmetry().insert(elem2);
        ctrl2.req_symmetry().insert(elem2);
    }

    //  Fill in random data

    bto_random<3, T>().perform(bt1);
    bto_random<3, T>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    //  Compute the dot product

    permutation<3> perm1, perm2;
    perm1.permute(0, 1).permute(1, 2);
    perm2.permute(1, 2).permute(0, 1);
    T d = bto_dotprod<3, T>(bt1, perm1, bt2, perm2).calculate();

    //  Compute the reference

    dense_tensor<3, T, allocator_t> t1(dims), t2(dims);
    to_btconv<3, T>(bt1).perform(t1);
    to_btconv<3, T>(bt2).perform(t2);
    T d_ref = to_dotprod<3, T>(t1, perm1, t2, perm2).calculate();

    //  Compare

    if(std::abs(d - d_ref) > std::abs(d_ref) * k_thresh) {
        std::ostringstream ss;
        ss << "Result does not match reference: " << d << " vs. "
            << d_ref << " (ref), " << d - d_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


template<typename T>
void bto_dotprod_test_x<T>::test_13b() throw(libtest::test_exception) {

    //  3-dim tensor, four blocks in each dimension,
    //  permutational anti-symmetry

    static const char testname[] = "bto_dotprod_test_x<T>::test_13b()";

    typedef allocator<T> allocator_t;

    try {

    index<3> i1, i2;
    i2[0] = 9; i2[1] = 9; i2[2] = 9;
    dimensions<3> dims(index_range<3>(i1, i2));
    block_index_space<3> bis(dims);
    mask<3> m;
    m[0] = true; m[1] = true; m[2] = true;
    bis.split(m, 2);
    bis.split(m, 5);
    bis.split(m, 7);
    block_tensor<3, T, allocator_t> bt1(bis), bt2(bis);

    //  Set up symmetry

    {
        block_tensor_ctrl<3, T> ctrl1(bt1), ctrl2(bt2);

        scalar_transf<T> tr0, tr1(-1.);
        se_perm<3, T> elem1(permutation<3>().permute(0, 1), tr1);
        se_perm<3, T> elem2(permutation<3>().permute(1, 2), tr1);

        ctrl1.req_symmetry().insert(elem1);
        ctrl1.req_symmetry().insert(elem2);
        ctrl2.req_symmetry().insert(elem2);
    }

    //  Fill in random data

    bto_random<3, T>().perform(bt1);
    bto_random<3, T>().perform(bt2);
    bt1.set_immutable();
    bt2.set_immutable();

    //  Compute the dot product

    permutation<3> perm1, perm2;
    perm1.permute(0, 1).permute(1, 2);
    perm2.permute(1, 2).permute(0, 1);
    T d = bto_dotprod<3, T>(bt2, perm2, bt1, perm1).calculate();

    //  Compute the reference

    dense_tensor<3, T, allocator_t> t1(dims), t2(dims);
    to_btconv<3, T>(bt1).perform(t1);
    to_btconv<3, T>(bt2).perform(t2);
    T d_ref = to_dotprod<3, T>(t1, perm1, t2, perm2).calculate();

    //  Compare

    if(std::abs(d - d_ref) > std::abs(d_ref) * k_thresh) {
        std::ostringstream ss;
        ss << "Result does not match reference: " << d << " vs. "
            << d_ref << " (ref), " << d - d_ref << " (diff).";
        fail_test(testname, __FILE__, __LINE__, ss.str().c_str());
    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}


} // namespace libtensor
