#include "defs.h"

namespace libtensor {

const char *version =
#ifdef LIBTENSOR_SVN_REV
	LIBTENSOR_SVN_REV;
#else
	"unknown";
#endif

const char *g_ns = "libtensor";

namespace labeled_btensor_expr {

const char *g_ns = "libtensor::labeled_btensor_expr";

}

}

