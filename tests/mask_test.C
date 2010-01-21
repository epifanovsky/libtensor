#include <libtensor/core/mask.h>
#include "mask_test.h"

namespace libtensor {

void mask_test::perform() throw(libtest::test_exception) {

	mask<2> msk1;
	mask<2> msk2(msk1);
}

}
