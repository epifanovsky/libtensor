#include <cmath>
#include <ctime>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_set_elem.h>
#include "../compare_ref.h"
#include "../test_utils.h"

using namespace libtensor;


int test_1() {

    static const char testname[] = "tod_set_elem_test::test_1()";

    typedef allocator<double> allocator_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 3; i2[1] = 4;
    dimensions<2> dims(index_range<2>(i1, i2));
    dense_tensor<2, double, allocator_t> t(dims), t_ref(dims);

    {
    dense_tensor_ctrl<2, double> tc(t), tc_ref(t_ref);

    // Fill in random data
    //
    double *d = tc.req_dataptr();
    double *d_ref = tc_ref.req_dataptr();
    size_t sz = dims.get_size();
    for(size_t i = 0; i < sz; i++) d_ref[i] = d[i] = drand48();
    tc.ret_dataptr(d); d = 0;
    tc_ref.ret_dataptr(d_ref); d_ref = 0;

    // Test [0,0]
    //
    libtensor::index<2> i00;
    abs_index<2> ai00(i00, dims);
    double q = drand48();
    d_ref = tc_ref.req_dataptr();
    d_ref[ai00.get_abs_index()] = q;
    tc_ref.ret_dataptr(d_ref); d_ref = 0;
    tod_set_elem<2>().perform(t, i00, q);
    compare_ref<2>::compare(testname, t, t_ref, 0.0);

    // Test [3, 2]
    //
    libtensor::index<2> i32; i32[0] = 3; i32[1] = 2;
    abs_index<2> ai32(i32, dims);
    q = drand48();
    d_ref = tc_ref.req_dataptr();
    d_ref[ai32.get_abs_index()] = q;
    tc_ref.ret_dataptr(d_ref); d_ref = 0;
    tod_set_elem<2>().perform(t, i32, q);
    compare_ref<2>::compare(testname, t, t_ref, 0.0);
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    srand48(time(0));

    return

    test_1() |

    0;
}


