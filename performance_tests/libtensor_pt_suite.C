#include <libtensor.h>
#include "libtensor_pt_suite.h"

namespace libtensor {


libtensor_pt_suite::libtensor_pt_suite()
	: performance_test_suite("libtensor_performance_test")
{
	add_tests("tod_add",
			"Size:     8x8x8x8, Repeats: 1000000", m_tod_add_ptsc1);
	add_tests("tod_add",
			"Size: 16x16x16x16, Repeats:   60000", m_tod_add_ptsc2);
	add_tests("tod_add",
			"Size: 32x32x32x32, Repeats:    4000", m_tod_add_ptsc3);
	add_tests("tod_contract2",
			"Size:     8x8x8x8, Repeats:   10000", m_tod_contract2_ptsc1);
	add_tests("tod_contract2",
			"Size: 16x16x16x16, Repeats:     600", m_tod_contract2_ptsc2);
	add_tests("tod_contract2",
			"Size: 32x32x32x32, Repeats:      40", m_tod_contract2_ptsc3);
	add_tests("tod_copy",
			"Size:     8x8x8x8, Repeats: 1000000", m_tod_copy_ptsc1);
	add_tests("tod_copy",
			"Size: 16x16x16x16, Repeats:   60000", m_tod_copy_ptsc2);
	add_tests("tod_copy",
			"Size: 32x32x32x32, Repeats:    4000", m_tod_copy_ptsc3);
	add_tests("tod_dotprod",
			"Size:     8x8x8x8, Repeats: 1000000", m_tod_dotprod_ptsc1);
	add_tests("tod_dotprod",
			"Size: 16x16x16x16, Repeats:   60000", m_tod_dotprod_ptsc2);
	add_tests("tod_dotprod",
			"Size: 32x32x32x32, Repeats:    4000", m_tod_dotprod_ptsc3);
	add_tests("expressions",
			"Size: O=32, V= 64, BS=16; Repeats:  400", m_expression_tests_n16);
	add_tests("expressions",
			"Size: O=32, V= 64, BS= 8; Repeats:  200", m_expression_tests_n8);
	add_tests("expressions",
			"Size: O=32, V= 64, BS= 4; Repeats:  100", m_expression_tests_n4);
//	add_tests("expressions", "Size: O=32, V= 64, BS= 2; Repeats:  100",
//			m_expression_tests_n2);
	add_tests("expressions",
			"Size: O=32, V=256, BS=32; Repeats:  800", m_expression_tests_l32);
	add_tests("expressions",
			"Size: O=32, V=256, BS=16; Repeats:  400", m_expression_tests_l16);
	add_tests("expressions",
			"Size: O=32, V=256, BS= 8; Repeats:  200", m_expression_tests_l8);
	add_tests("expressions",
			"Size: O=32, V=256, BS= 4; Repeats:  100", m_expression_tests_l4);

}

} // namespace libtensor


