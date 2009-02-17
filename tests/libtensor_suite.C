#include <libtensor.h>
#include "libtensor_suite.h"

namespace libtensor {

libtensor_suite::libtensor_suite() {
	add_test("permutation", m_utf_permutation);
	add_test("permutation_lehmer", m_utf_permutation_lehmer);
}

}

