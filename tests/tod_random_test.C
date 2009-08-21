#include <libvmm.h>
#include "compare_ref.h"
#include "tod_random_test.h"

namespace libtensor {

typedef libvmm::std_allocator<double> allocator;
typedef tensor<3, double, allocator> tensor3;
typedef tensor_ctrl<3,double> tensor3_ctrl;

void tod_random_test::perform() throw(libtest::test_exception) 
{
	index<3> i3a, i3b; i3b[0]=10; i3b[1]=12; i3b[2]=11;
	index_range<3> ir3(i3a, i3b); dimensions<3> dims3(ir3);
	tensor3 ta3(dims3), tb3(dims3);
	
	tod_random<3> randr;
	bool test_ok=false;
	try {
		randr.perform(ta3);
		randr.perform(tb3); 	
	
		compare_ref<3>::compare("tod_random_test",ta3,tb3,0.0);		
	} catch ( libtest::test_exception& e ) {
		test_ok=true;
	} catch ( exception& e ) {
		fail_test("tod_random_test", __FILE__, __LINE__, e.what());
	}
	if ( ! test_ok ) 
		fail_test("tod_random_test", __FILE__, __LINE__, 
			"Two identical random number sequences.");
			
			
	randr.perform(ta3,2.0);
	tensor_ctrl<3,double> ctrla(ta3);
	const double *cptra=ctrla.req_const_dataptr();
	for (size_t i=0; i<ta3.get_dims().get_size(); i++ ) {
		if ( (*cptra<0.0) || (*cptra>=2.0) )
			fail_test("tod_random_test<N>",__FILE__,__LINE__,
				"Random numbers outside specified interval");
	} 	
}

} // namespace libtensor

