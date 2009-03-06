#include "defs.h"

namespace libtensor {

const char *version =
#ifdef LIBTENSOR_SVN_REV
	LIBTENSOR_SVN_REV;
#else
	"unknown";
#endif

}

