#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/print_dimensions.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_set_diag.h>
#include "../compare_ref.h"
#include "../test_utils.h"

using namespace libtensor;


template<size_t N>
int run_test1(const dimensions<N> &dims, double d, bool zero) {

    std::ostringstream tnss;
    tnss << "tod_set_diag_test::run_test1(" << dims << ", " << d << ","
            << (zero ? "t" : "f") << ")";

    try {

    dense_tensor<N, double, allocator> t(dims), t_ref(dims);

    {
    dense_tensor_ctrl<N, double> ctrl(t), ctrl_ref(t_ref);

    double *p = ctrl.req_dataptr();
    double *p_ref = ctrl_ref.req_dataptr();

    // Fill in random data & prepare the reference

    abs_index<N> ai(dims);
    do {
        size_t n = ai.get_index().at(0);
        bool diag = true;
        for(size_t j = 1; j < N; j++) {
            if(ai.get_index().at(j) != n) {
                diag = false;
                break;
            }
        }
        if(diag) {
            p[ai.get_abs_index()] = drand48();
            if (zero)
                p_ref[ai.get_abs_index()] = d;
            else
                p_ref[ai.get_abs_index()] = p[ai.get_abs_index()] + d;
        } else {
            p[ai.get_abs_index()] = p_ref[ai.get_abs_index()] = drand48();
        }
    } while(ai.inc());

    ctrl.ret_dataptr(p); p = NULL;
    ctrl_ref.ret_dataptr(p_ref); p_ref = NULL;
    t_ref.set_immutable();
    }

    // Run the operation

    to_set_diag<N, double>(d).perform(zero, t);

    // Compare against the reference

    compare_ref<N>::compare(tnss.str().c_str(), t, t_ref, 0.0);

    } catch(exception &e) {
        return fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


template<size_t N>
int run_test2(
    const dimensions<N> &dims, const sequence<N, size_t> &msk, double d,
    bool zero) {

    std::ostringstream tnss;
    tnss << "tod_set_diag_test::run_test2(" << dims << ", [";
    for (size_t i = 0; i < N - 1; i++) tnss << msk[i] << ", ";
    tnss << msk[N - 1] << "], " << d << ")";

    try {

    dense_tensor<N, double, allocator> t(dims), t_ref(dims);

    {
    dense_tensor_ctrl<N, double> ctrl(t), ctrl_ref(t_ref);

    double *p = ctrl.req_dataptr();
    double *p_ref = ctrl_ref.req_dataptr();

    // Fill in random data & prepare the reference

    abs_index<N> ai(dims);
    do {
        bool diag = true;
        for (size_t j = 0; j < N; j++) {
            if (msk[j] == 0) continue;

            size_t n = ai.get_index().at(j);
            for (size_t k = j + 1; k < N; k++) {
                if (msk[j] == msk[k] && ai.get_index().at(k) != n) {
                    diag = false;
                    break;
                }
            }
            if (! diag) break;
        }
        if(diag) {
            p[ai.get_abs_index()] = drand48();
            if (zero) p_ref[ai.get_abs_index()] = d;
            else p_ref[ai.get_abs_index()] = p[ai.get_abs_index()] + d;
        } else {
            p[ai.get_abs_index()] = p_ref[ai.get_abs_index()] = drand48();
        }
    } while(ai.inc());

    ctrl.ret_dataptr(p); p = NULL;
    ctrl_ref.ret_dataptr(p_ref); p_ref = NULL;
    t_ref.set_immutable();
    }

    // Run the operation

    to_set_diag<N, double>(msk, d).perform(zero, t);

    // Compare against the reference

    compare_ref<N>::compare(tnss.str().c_str(), t, t_ref, 0.0);

    } catch(exception &e) {
        return fail_test(tnss.str().c_str(), __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    srand48(time(NULL));

    libtensor::index<2> i2a, i2b;
    i2b[0] = 10; i2b[1] = 10;
    dimensions<2> dims2_10(index_range<2>(i2a, i2b));

    libtensor::index<4> i4a, i4b;
    i4b[0] = 5; i4b[1] = 10; i4b[2] = 5; i4b[3] = 10;
    dimensions<4> dims4(index_range<4>(i4a, i4b));
    sequence<4, size_t> m1(0), m2(0);
    m1[1] = 1; m1[3] = 1;
    m2[0] = 1; m2[1] = 2; m2[2] = 1; m2[3] = 2;

    return

    run_test1(dims2_10, 0.0, true) |
    run_test1(dims2_10, 0.0, false) |
    run_test1(dims2_10, 11.5, true) |
    run_test1(dims2_10, 11.5, false) |

    run_test2(dims4, m1, 0.0, true) |
    run_test2(dims4, m1, 0.0, false) |
    run_test2(dims4, m1, 5.0, true) |
    run_test2(dims4, m1, 5.0, false) |
    run_test2(dims4, m2, 0.0, true) |
    run_test2(dims4, m2, 0.0, false) |
    run_test2(dims4, m2, 5.0, true) |
    run_test2(dims4, m2, 5.0, false) |

    0;
}


