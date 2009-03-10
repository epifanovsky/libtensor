#include <cstdio>
#include "exception.h"

namespace libtensor {

void throw_exc(const char *clazz, const char *method, const char *error)
	throw(exception) {
	char s[1024];
	snprintf(s, 1024, "[%s::%s] %s.", clazz, method, error);
	throw exception(s);
}

} // namespace libtensor

