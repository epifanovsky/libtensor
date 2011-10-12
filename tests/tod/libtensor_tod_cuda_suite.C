#include <libtensor/libtensor.h>
#include "libtensor_tod_cuda_suite.h"

namespace libtensor {

libtensor_tod_cuda_suite::libtensor_tod_cuda_suite() : libtest::test_suite("libtensor") {

	add_test("tod_cuda_copy", m_utf_tod_cuda_copy);
}

}

