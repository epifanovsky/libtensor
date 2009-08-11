#include <libtensor.h>
#include "libtensor_performance_suite.h"

namespace libtensor {

libtensor_performance_suite::libtensor_performance_suite() : libtest::test_suite("libtensor_performance") {
	add_test("expression_p", m_utf_expression_p);
}

}

