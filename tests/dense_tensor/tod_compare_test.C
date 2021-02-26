#include <cmath>
#include <cstdlib>
#include <ctime>
#include <libtensor/core/allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_compare.h>
#include "../test_utils.h"

using namespace libtensor;


int test_exc() {

    typedef dense_tensor<4, double, allocator> tensor4;

    libtensor::index<4> i1, i2, i3;
    i2[0]=2; i2[1]=2; i2[2]=2; i2[3]=2;
    i3[0]=3; i3[1]=3; i3[2]=3; i3[3]=3;
    index_range<4> ir1(i1,i2), ir2(i1,i3);
    dimensions<4> dim1(ir1), dim2(ir2);
    tensor4 t1(dim1), t2(dim2);

    bool ok = false;
    try {
        to_compare<4, double> tc(t1, t2, 0);
    } catch(exception &e) {
        ok = true;
    }

    if(!ok) {
        return fail_test("tod_compare_test::test_exc()", __FILE__, __LINE__,
        "Expected an exception with heterogeneous arguments");
    }

    return 0;
}

int test_operation(const dimensions<4> &dim,
    const libtensor::index<4> &idx) {

    typedef dense_tensor<4, double, allocator> tensor4;
    typedef dense_tensor_ctrl<4,double> tensor4_ctrl;

    tensor4 t1(dim), t2(dim);

    double diff1, diff2;
    size_t diffptr;
    {
        tensor4_ctrl tctrl1(t1), tctrl2(t2);

        double *p1 = tctrl1.req_dataptr();
        double *p2 = tctrl2.req_dataptr();

        size_t sz = dim.get_size();
        for(size_t i=0; i<sz; i++) {
        p2[i] = p1[i] = drand48();
        }
        diffptr = abs_index<4>::get_abs_index(idx, dim);
        p2[diffptr] += 1e-6;
        diff1 = p1[diffptr];
        diff2 = p2[diffptr];

        tctrl1.ret_dataptr(p1);
        tctrl2.ret_dataptr(p2);
    }

    to_compare<4, double> op1(t1, t2, 1e-7);
    if(op1.compare()) {
        return fail_test("tod_compare_test::test_operation()", __FILE__,
        __LINE__, "tod_compare failed to find the difference");
    }
    if(abs_index<4>::get_abs_index(op1.get_diff_index(), dim) != diffptr) {
        return fail_test("tod_compare_test::test_operation()", __FILE__,
        __LINE__, "tod_compare returned an incorrect index");
    }
    if(op1.get_diff_elem_1() != diff1 || op1.get_diff_elem_2() != diff2) {
        return fail_test("tod_compare_test::test_operation()", __FILE__,
        __LINE__, "tod_compare returned an incorrect "
        "element value");
    }

    to_compare<4, double> op2(t1, t2, 1e-5);
    if(!op2.compare()) {
        return fail_test("tod_compare_test::test_operation()", __FILE__,
        __LINE__, "tod_compare found a difference below "
        "the threshold");
    }

    return 0;
}


/** \test Tests to_compare<0, double>
 **/
int test_0() {

    static const char testname[] = "tod_compare_test::test_0()";

    try {

    libtensor::index<0> i1, i2;
    dimensions<0> dims(index_range<0>(i1, i2));
    dense_tensor<0, double, allocator> t1(dims), t2(dims), t3(dims);

    {
        dense_tensor_ctrl<0, double> tc1(t1), tc2(t2), tc3(t3);

        double *p1 = tc1.req_dataptr();
        double *p2 = tc2.req_dataptr();
        double *p3 = tc3.req_dataptr();
        *p1 = 1.0; *p2 = 1.0; *p3 = -2.5;
        tc1.ret_dataptr(p1); p1 = 0;
        tc2.ret_dataptr(p2); p2 = 0;
        tc3.ret_dataptr(p3); p3 = 0;
    }

    to_compare<0, double> comp1(t1, t2, 0.0);
    if(!comp1.compare()) {
        return fail_test(testname, __FILE__, __LINE__, "!comp1.compare()");
    }
    to_compare<0, double> comp2(t1, t3, 0.0);
    if(comp2.compare()) {
        return fail_test(testname, __FILE__, __LINE__, "comp2.compare()");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


/** \test Tests to_compare<2, double>
 **/
int test_1() {

    static const char testname[] = "tod_compare_test::test_1()";

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 5; i2[1] = 5;
    dimensions<2> dims(index_range<2>(i1, i2));
    size_t sz = dims.get_size();
    dense_tensor<2, double, allocator> t1(dims), t2(dims);

    {
        dense_tensor_ctrl<2, double> tc1(t1), tc2(t2);

        double *p1 = tc1.req_dataptr();
        double *p2 = tc2.req_dataptr();

        for(size_t i = 0; i < sz; i++) {
        p1[i] = 100.0;
        p2[i] = 100.0;
        if(i % 3 == 0) p2[i] += 1e-9;
        if(i % 3 == 1) p2[i] -= 1e-9;
        }

        tc1.ret_dataptr(p1); p1 = 0;
        tc2.ret_dataptr(p2); p2 = 0;
    }

    to_compare<2, double> comp1(t1, t2, 1e-10);
    if(!comp1.compare()) {
        return fail_test(testname, __FILE__, __LINE__, "!comp1.compare()");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    srand48(time(NULL));

    libtensor::index<4> i1, i2; i2[0]=2; i2[1]=3; i2[2]=4; i2[3]=5;
    libtensor::index<4> idiff; idiff[0]=0; idiff[1]=1; idiff[2]=2; idiff[3]=3;
    index_range<4> ir(i1,i2);
    dimensions<4> dim(ir);

    return

    test_exc() |
    test_operation(dim, idiff) |

    test_0() |
    test_1() |

    0;
}

