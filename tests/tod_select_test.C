#include <cmath>
#include <ctime>
#include <cstdlib>
#include <sstream>
#include <libvmm/std_allocator.h>
#include <libtensor/core/tensor.h>
#include <libtensor/tod/tod_select.h>
#include "tod_select_test.h"


namespace libtensor {


void tod_select_test::perform() throw(libtest::test_exception) {

	srand48(time(0));

	test_1();
}


void tod_select_test::test_1() throw(libtest::test_exception) {

	static const char *testname = "tod_select_test::test_1()";

	typedef libvmm::std_allocator<double> allocator_t;

	try {

	index<2> i1, i2;
	i2[0] = 3; i2[1] = 4;
	dimensions<2> dims(index_range<2>(i1, i2));
	tensor<2, double, allocator_t> t(dims);

	size_t sz;
	{
	tensor_ctrl<2, double> tc(t);

	//	Fill in random data
	//
	double *d = tc.req_dataptr();
	sz = dims.get_size();
	for(size_t i = 0; i < sz; i++) d[i] = drand48();
	tc.ret_dataptr(d); d = 0;

	// compare4absmin
	compare4absmin cmp;
	tod_select<2> tsel(cmp);
	typedef tod_select<2>::list_t list_t;
	list_t li_am;
	tsel.perform(t,li_am,4);

	list_t::iterator it=li_am.begin();
	const double *cd = tc.req_const_dataptr();
	while ( it != li_am.end() ) {
		for (size_t i=0; i<sz; i++) {
			if ( cd[i]==0.0 ) continue;

			if ( cmp(cd[i],it->value) ) {
				list_t::iterator it2=li_am.begin();
				bool ok=false;
				while ( it2 != it ) {
					if (cd[i] == it2->value &&
							i == dims.abs_index(it2->idx))
						ok=true;
					it2++;
				}
				if (!ok) {
					std::ostringstream oss;
					index<2> idx;
					dims.abs_index(i,idx);
					oss << "Unsorted list at element (" << it->idx << ", "
							<< it->value << "). Found in tensor at "
							<< idx << ", value = " << cd[i] << ".";
					fail_test(testname,__FILE__,__LINE__,oss.str().c_str());
				}
			}
		}
		it++;
	}
	}

	} catch(exception &e) {
		fail_test(testname, __FILE__, __LINE__, e.what());
	}
}


} // namespace libtensor
