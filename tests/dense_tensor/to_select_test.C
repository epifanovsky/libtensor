#include <cmath>
#include <ctime>
#include <cstdlib>
#include <sstream>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_select.h>
#include <libtensor/dense_tensor/impl/to_select_impl.h>
#include "to_select_test.h"


namespace libtensor {

void to_select_test::perform() throw(libtest::test_exception) {
    std::cout << "Testing to_select_test_x<double>    ";
    to_select_test_x<double> t_double;
    t_double.perform();
    std::cout << "Testing to_select_test_x<float>    ";
    to_select_test_x<float> t_float;
    t_float.perform();
}

template<typename T>
void to_select_test_x<T>::perform() throw(libtest::test_exception) {

    srand48(time(0));

    test_1<compare4absmax<T> >(4, 1.0);
    test_1<compare4absmax<T> >(4, -2.0);
    test_1<compare4min<T> >(4, 0.5);
    test_1<compare4min<T> >(4, -1.0);

    test_2<compare4absmin<T> >(4, 1.0);
    test_2<compare4absmin<T> >(4, -0.5);
    test_2<compare4max<T> >(4, 2.0);
    test_2<compare4max<T> >(4, -1.0);

}

template<typename T>
template<typename ComparePolicy>
void to_select_test_x<T>::test_1(size_t n, T c)
        throw(libtest::test_exception) {

    static const char *testname = "to_select_test_x<T>::test_1()";

    typedef allocator<T> allocator_t;
    typedef typename to_select<2, T, ComparePolicy>::list_type list_type;

    try {

    index<2> i1, i2;
    i2[0] = 3; i2[1] = 4;
    dimensions<2> dims(index_range<2>(i1, i2));
    dense_tensor<2, T, allocator_t> t(dims);

    size_t sz;
    sz = dims.get_size();

    {
    //
    // Fill in random data
    //
    dense_tensor_ctrl<2, T> tc(t);
    T *d = tc.req_dataptr();

    for(size_t i = 0; i < sz; i++) d[i] = drand48();

    tc.ret_dataptr(d); d = 0;

    }

    // Perform the operation
    ComparePolicy cmp;
    list_type li;
    to_select<2, T, ComparePolicy> tsel(t, c, cmp);
    tsel.perform(li, n);

    { // Check the resulting list

    dense_tensor_ctrl<2, T> tc(t);
    const T *cd = tc.req_const_dataptr();
    // Loop over all list elements
    for (typename list_type::const_iterator it = li.begin();
        it != li.end(); it++) {

        // Loop over all data elements in tensor
        for (size_t i = 0; i < sz; i++) {

        if (cd[i] == 0.0) continue;

        T val = cd[i] * c;
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
                fail_test(testname, __FILE__, __LINE__,
                        oss.str().c_str());
            }
        }
        } // for i
    } // for it

    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

template<typename T>
template<typename ComparePolicy>
void to_select_test_x<T>::test_2(size_t n, T c)
        throw(libtest::test_exception) {

    static const char *testname = "to_select_test_x<T>::test_2()";

    typedef allocator<T> allocator_t;
    typedef typename to_select<3, T, ComparePolicy>::list_type list_type;

    try {

    permutation<3> perm; perm.permute(0, 1).permute(1, 2);
    permutation<3> pinv(perm, true);
    index<3> i1, i2;
    i2[0] = 3; i2[1] = 4; i2[2] = 2;
    dimensions<3> dims(index_range<3>(i1, i2));
    dense_tensor<3, T, allocator_t> t(dims);

    size_t sz;
    sz = dims.get_size();

    {
    //
    // Fill in random data
    //
    dense_tensor_ctrl<3, T> tc(t);
    T *d = tc.req_dataptr();

    for(size_t i = 0; i < sz; i++) d[i] = drand48();

    tc.ret_dataptr(d); d = 0;

    }

    // Perform the operation
    ComparePolicy cmp;
    list_type li;
    to_select<3, T, ComparePolicy> tsel(t, perm, c, cmp);
    tsel.perform(li, n);

    { // Check the resulting list

    dense_tensor_ctrl<3, T> tc(t);
    const T *cd = tc.req_const_dataptr();
    // Loop over all list elements
    for (typename list_type::const_iterator it = li.begin();
        it != li.end(); it++) {

        // Loop over all data elements in tensor
        for (size_t i = 0; i < sz; i++) {

        if (cd[i] == 0.0) continue;

        T val = cd[i] * c;
        if (cmp(val, it->get_value())) {

            bool ok = false;
            for (typename list_type::const_iterator it2 = li.begin();
                    it2 != it; it2++) {

                index<3> idx(it2->get_index());
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
                fail_test(testname, __FILE__, __LINE__,
                        oss.str().c_str());
            }
        }
        } // for i
    } // for it

    }

    } catch(exception &e) {
        fail_test(testname, __FILE__, __LINE__, e.what());
    }
}

} // namespace libtensor
