#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/block_tensor/block_factory.h>
#include <libtensor/gen_block_tensor/impl/block_map_impl.h>
#include <libtensor/block_tensor/block_tensor_i_traits.h>
#include "block_map_test.h"

namespace libtensor {


namespace block_map_test_ns {

struct bt_traits {

    typedef double element_type;
    typedef std_allocator<double> allocator_type;
    typedef block_tensor_i_traits<double> bti_traits;

    template<size_t N>
    struct block_type {
        typedef dense_tensor< N, double, std_allocator<double> > type;
    };

    template<size_t N>
    struct block_factory_type {
        typedef block_factory<N, double, typename block_type<N>::type> type;
    };

};

} // namespace block_map_test_ns
using namespace block_map_test_ns;


void block_map_test::perform() throw(libtest::test_exception) {

	test_create();
	test_immutable();
}


void block_map_test::test_create() throw(libtest::test_exception) {

	static const char *testname = "block_map_test::test_create()";

	try {

    index<2> i1, i2;
    i2[0] = 8; i2[1] = 12;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m01, m10;
    m10[0] = true; m01[1] = true;
    bis.split(m10, 4);
    bis.split(m01, 6);

    index<2> i00, i11;
    i11[0] = 1; i11[1] = 1;

    i2[0] = 3; i2[1] = 5;
    dimensions<2> dims1(index_range<2>(i1, i2));
    i2[0] = 4; i2[1] = 6;
    dimensions<2> dims2(index_range<2>(i1, i2));

	block_map<2, bt_traits> map(bis);

	if(map.contains(i00)) {
		fail_test(testname, __FILE__, __LINE__,
			"Nonexisting block [0,0] reported to be found (1)");
	}
	if(map.contains(i11)) {
		fail_test(testname, __FILE__, __LINE__,
			"Nonexisting block [1,1] reported to be found (1)");
	}

	map.create(i00);
	if(!map.contains(i00)) {
		fail_test(testname, __FILE__, __LINE__,
			"Existing block [0,0] cannot be found (2)");
	}
	if(map.contains(i11)) {
		fail_test(testname, __FILE__, __LINE__,
			"Nonexisting block [1,1] reported to be found (2)");
	}

	dense_tensor_i<2, double> &ta1 = map.get(i00);
	if(!ta1.get_dims().equals(dims1)) {
		fail_test(testname, __FILE__, __LINE__,
			"Block [0,0] has incorrect dimensions (2)");
	}

	map.create(i11);
	if(!map.contains(i00)) {
		fail_test(testname, __FILE__, __LINE__,
			"Existing block [0,0] cannot be found (3)");
	}
	if(!map.contains(i11)) {
		fail_test(testname, __FILE__, __LINE__,
			"Existing block [1,1] cannot be found (3)");
	}

	dense_tensor_i<2, double> &tb1 = map.get(i00);
	dense_tensor_i<2, double> &tb2 = map.get(i11);
	if(!tb1.get_dims().equals(dims1)) {
		fail_test(testname, __FILE__, __LINE__,
			"Block [0,0] has incorrect dimensions (3)");
	}
	if(!tb2.get_dims().equals(dims2)) {
		fail_test(testname, __FILE__, __LINE__,
			"Block [1,1] has incorrect dimensions (3)");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}

void block_map_test::test_immutable() throw(libtest::test_exception) {

	static const char *testname = "block_map_test::test_immutable()";

    typedef std_allocator<double> allocator_t;
    typedef dense_tensor<2, double, allocator_t> tensor_t;

	try {

    index<2> i1, i2;
    i2[0] = 6; i2[1] = 10;
    dimensions<2> dims(index_range<2>(i1, i2));
    block_index_space<2> bis(dims);
    mask<2> m01, m10;
    m10[0] = true; m01[1] = true;
    bis.split(m10, 3);
    bis.split(m01, 5);

    index<2> i00, i11;
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
		fail_test(testname, __FILE__, __LINE__,
			"Block 0 is not immutable.");
	}
	if(!tb2.is_immutable()) {
		fail_test(testname, __FILE__, __LINE__,
			"Block 2 is not immutable.");
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
