#include <cmath>
#include <ctime>
#include <cstdlib>
#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/tod_select.h>
#include <libtensor/dense_tensor/impl/tod_select_impl.h>
#include "../test_utils.h"


using namespace libtensor;


template<typename ComparePolicy>
int test_1(size_t n, double c) {

    static const char testname[] = "tod_select_test::test_1()";

    typedef allocator<double> allocator_t;
    typedef typename tod_select<2, ComparePolicy>::list_type list_type;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 3; i2[1] = 4;
    dimensions<2> dims(index_range<2>(i1, i2));
    dense_tensor<2, double, allocator_t> t(dims);

    size_t sz;
    sz = dims.get_size();

    {
    //
    // Fill in random data
    //
    dense_tensor_ctrl<2, double> tc(t);
    double *d = tc.req_dataptr();

    for(size_t i = 0; i < sz; i++) d[i] = drand48();

    tc.ret_dataptr(d); d = 0;

    }

    // Perform the operation
    ComparePolicy cmp;
    list_type li;
    tod_select<2, ComparePolicy> tsel(t, c, cmp);
    tsel.perform(li, n);

    { // Check the resulting list

    dense_tensor_ctrl<2, double> tc(t);
    const double *cd = tc.req_const_dataptr();
    // Loop over all list elements
    for (typename list_type::const_iterator it = li.begin();
        it != li.end(); it++) {

        // Loop over all data elements in tensor
        for (size_t i = 0; i < sz; i++) {

        if (cd[i] == 0.0) continue;

        double val = cd[i] * c;
        if (cmp(val, it->get_value())) {

            bool ok = false;
            for (typename list_type::const_iterator it2 = li.begin();
                    it2 != it; it2++) {

                abs_index<2> aidx(it2->get_index(), dims);
                if (val == it2->get_value() &&
                        i == aidx.get_abs_index()) {
                    ok = true; break;
                }
            }

            if (! ok) {
                std::ostringstream oss;
                abs_index<2> aidx(i, dims);
                oss << "Unsorted list at element (" << it->get_index() << ", "
                        << it->get_value() << "). Found in tensor at "
                        << aidx.get_index() << ", value = " << cd[i] << ".";
                return fail_test(testname, __FILE__, __LINE__,
                        oss.str().c_str());
            }
        }
        } // for i
    } // for it

    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


template<typename ComparePolicy>
int test_2(size_t n, double c) {

    static const char testname[] = "tod_select_test::test_2()";

    typedef allocator<double> allocator_t;
    typedef typename tod_select<3, ComparePolicy>::list_type list_type;

    try {

    permutation<3> perm; perm.permute(0, 1).permute(1, 2);
    permutation<3> pinv(perm, true);
    libtensor::index<3> i1, i2;
    i2[0] = 3; i2[1] = 4; i2[2] = 2;
    dimensions<3> dims(index_range<3>(i1, i2));
    dense_tensor<3, double, allocator_t> t(dims);

    size_t sz;
    sz = dims.get_size();

    {
    //
    // Fill in random data
    //
    dense_tensor_ctrl<3, double> tc(t);
    double *d = tc.req_dataptr();

    for(size_t i = 0; i < sz; i++) d[i] = drand48();

    tc.ret_dataptr(d); d = 0;

    }

    // Perform the operation
    ComparePolicy cmp;
    list_type li;
    tod_select<3, ComparePolicy> tsel(t, perm, c, cmp);
    tsel.perform(li, n);

    { // Check the resulting list

    dense_tensor_ctrl<3, double> tc(t);
    const double *cd = tc.req_const_dataptr();
    // Loop over all list elements
    for (typename list_type::const_iterator it = li.begin();
        it != li.end(); it++) {

        // Loop over all data elements in tensor
        for (size_t i = 0; i < sz; i++) {

        if (cd[i] == 0.0) continue;

        double val = cd[i] * c;
        if (cmp(val, it->get_value())) {

            bool ok = false;
            for (typename list_type::const_iterator it2 = li.begin();
                    it2 != it; it2++) {

                libtensor::index<3> idx(it2->get_index());
                idx.permute(pinv);
                abs_index<3> aidx(idx, dims);
                if (val == it2->get_value() &&
                        i == aidx.get_abs_index()) {
                    ok = true; break;
                }
            }

            if (! ok) {
                std::ostringstream oss;
                abs_index<3> aidx(i, dims);
                oss << "Unsorted list at element (" << it->get_index()
                                << ", " << it->get_value()
                                << "). Found in tensor at " << aidx.get_index()
                                << ", value = " << cd[i] << ".";
                return fail_test(testname, __FILE__, __LINE__,
                        oss.str().c_str());
            }
        }
        } // for i
    } // for it

    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}

int main() {

    srand48(time(0));

    return

    test_1<compare4absmax>(4, 1.0) |
    test_1<compare4absmax>(4, -2.0) |
    test_1<compare4min>(4, 0.5) |
    test_1<compare4min>(4, -1.0) |

    test_2<compare4absmin>(4, 1.0) |
    test_2<compare4absmin>(4, -0.5) |
    test_2<compare4max>(4, 2.0) |
    test_2<compare4max>(4, -1.0) |

    0;
}

