#include <algorithm>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/block_tensor/block_factory.h>
#include <libtensor/gen_block_tensor/impl/block_map_impl.h>
#include <libtensor/block_tensor/block_tensor_i_traits.h>
#include "../test_utils.h"

using namespace libtensor;

namespace {

struct bt_traits {

    typedef double element_type;
    typedef allocator<double> allocator_type;
    typedef block_tensor_i_traits<double> bti_traits;

    template<size_t N>
    struct block_type {
        typedef dense_tensor< N, double, allocator<double> > type;
    };

    template<size_t N>
    struct block_factory_type {
        typedef block_factory<N, double, typename block_type<N>::type> type;
    };

};

} // unnamed namespace


int test_create() {

    static const char testname[] = "block_map_test::test_create()";

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 8; i2[1] = 12;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m01, m10;
    m10[0] = true; m01[1] = true;
    bis.split(m10, 4);
    bis.split(m01, 6);

    libtensor::index<2> i00, i11;
    i11[0] = 1; i11[1] = 1;

    i2[0] = 3; i2[1] = 5;
    dimensions<2> dims1(index_range<2>(i1, i2));
    i2[0] = 4; i2[1] = 6;
    dimensions<2> dims2(index_range<2>(i1, i2));

    block_map<2, bt_traits> map(bis);

    if(map.contains(i00)) {
        return fail_test(testname, __FILE__, __LINE__,
            "Nonexisting block [0,0] reported to be found (1)");
    }
    if(map.contains(i11)) {
        return fail_test(testname, __FILE__, __LINE__,
            "Nonexisting block [1,1] reported to be found (1)");
    }

    map.create(i00);
    if(!map.contains(i00)) {
        return fail_test(testname, __FILE__, __LINE__,
            "Existing block [0,0] cannot be found (2)");
    }
    if(map.contains(i11)) {
        return fail_test(testname, __FILE__, __LINE__,
            "Nonexisting block [1,1] reported to be found (2)");
    }

    dense_tensor_i<2, double> &ta1 = map.get(i00);
    if(!ta1.get_dims().equals(dims1)) {
        return fail_test(testname, __FILE__, __LINE__,
            "Block [0,0] has incorrect dimensions (2)");
    }

    map.create(i11);
    if(!map.contains(i00)) {
        return fail_test(testname, __FILE__, __LINE__,
            "Existing block [0,0] cannot be found (3)");
    }
    if(!map.contains(i11)) {
        return fail_test(testname, __FILE__, __LINE__,
            "Existing block [1,1] cannot be found (3)");
    }

    dense_tensor_i<2, double> &tb1 = map.get(i00);
    dense_tensor_i<2, double> &tb2 = map.get(i11);
    if(!tb1.get_dims().equals(dims1)) {
        return fail_test(testname, __FILE__, __LINE__,
            "Block [0,0] has incorrect dimensions (3)");
    }
    if(!tb2.get_dims().equals(dims2)) {
        return fail_test(testname, __FILE__, __LINE__,
            "Block [1,1] has incorrect dimensions (3)");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_immutable() {

    static const char testname[] = "block_map_test::test_immutable()";

    typedef allocator<double> allocator_t;
    typedef dense_tensor<2, double, allocator_t> tensor_t;

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 6; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m01, m10;
    m10[0] = true; m01[1] = true;
    bis.split(m10, 3);
    bis.split(m01, 5);

    libtensor::index<2> i00, i11;
    i11[0] = 1; i11[1] = 1;

    i2[0] = 3; i2[1] = 5;
    dimensions<2> dims1(index_range<2>(i1, i2));
    i2[0] = 4; i2[1] = 6;
    dimensions<2> dims2(index_range<2>(i1, i2));

    block_map<2, bt_traits> map(bis);

    map.create(i00);
    map.create(i11);

    map.set_immutable();

    tensor_t &tb1 = map.get(i00);
    tensor_t &tb2 = map.get(i11);

    if(!tb1.is_immutable()) {
        return fail_test(testname, __FILE__, __LINE__,
            "Block 0 is not immutable.");
    }
    if(!tb2.is_immutable()) {
        return fail_test(testname, __FILE__, __LINE__,
            "Block 2 is not immutable.");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int test_get_all_1() {

    static const char testname[] = "block_map_test::test_get_all_1()";

    try {

    libtensor::index<2> i1, i2;
    i2[0] = 8; i2[1] = 12;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m01, m10;
    m10[0] = true; m01[1] = true;
    bis.split(m10, 4);
    bis.split(m01, 6);

    libtensor::index<2> i00, i11;
    i11[0] = 1; i11[1] = 1;

    i2[0] = 3; i2[1] = 5;
    dimensions<2> dims1(index_range<2>(i1, i2));
    i2[0] = 4; i2[1] = 6;
    dimensions<2> dims2(index_range<2>(i1, i2));

    std::vector<size_t> blst1, blst2;
    blst2.push_back(100);
    blst2.push_back(200);
    std::vector<size_t> blst3(blst1), blst4(blst2), blst5(blst1), blst6(blst2);

    block_map<2, bt_traits> map(bis);

    map.get_all(blst1);
    map.get_all(blst2);
    if(!blst1.empty()) {
        return fail_test(testname, __FILE__, __LINE__, "!blst1.empty()");
    }
    if(!blst2.empty()) {
        return fail_test(testname, __FILE__, __LINE__, "!blst2.empty()");
    }

    map.create(i00);
    map.get_all(blst3);
    map.get_all(blst4);
    if(blst3.size() != 1) {
        return fail_test(testname, __FILE__, __LINE__, "blst3.size() != 1");
    }
    if(blst3[0] != 0) {
        return fail_test(testname, __FILE__, __LINE__, "blst3[0] != 0");
    }
    if(blst4.size() != 1) {
        return fail_test(testname, __FILE__, __LINE__, "blst4.size() != 1");
    }
    if(blst4[0] != 0) {
        return fail_test(testname, __FILE__, __LINE__, "blst4[0] != 0");
    }

    map.create(i11);
    map.get_all(blst5);
    map.get_all(blst6);
    if(blst5.size() != 2) {
        return fail_test(testname, __FILE__, __LINE__, "blst5.size() != 2");
    }
    if(std::find(blst5.begin(), blst5.end(), 0) == blst5.end()) {
        return fail_test(testname, __FILE__, __LINE__, "blst5 doesn't contain [0,0]");
    }
    if(std::find(blst5.begin(), blst5.end(), 3) == blst5.end()) {
        return fail_test(testname, __FILE__, __LINE__, "blst5 doesn't contain [1,1]");
    }
    if(blst6.size() != 2) {
        return fail_test(testname, __FILE__, __LINE__, "blst6.size() != 2");
    }
    if(std::find(blst6.begin(), blst6.end(), 0) == blst6.end()) {
        return fail_test(testname, __FILE__, __LINE__, "blst6 doesn't contain [0,0]");
    }
    if(std::find(blst6.begin(), blst6.end(), 3) == blst6.end()) {
        return fail_test(testname, __FILE__, __LINE__, "blst6 doesn't contain [1,1]");
    }

    } catch(exception &e) {
        return fail_test(testname, __FILE__, __LINE__, e.what());
    }

    return 0;
}


int main() {

    return

    test_create() |
    test_immutable() |
    test_get_all_1() |

    0;
}

