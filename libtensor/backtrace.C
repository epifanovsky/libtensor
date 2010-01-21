#include <cstdlib>
#ifdef POSIX
#include <execinfo.h>
#endif
#include "backtrace.h"

namespace libtensor {

#ifdef POSIX
backtrace::backtrace() : m_avail(true) {

	const size_t nframes_max = 128;

	void *buf[nframes_max];
	char **lines;

	size_t nframes = ::backtrace(buf, nframes_max);

	lines = ::backtrace_symbols(buf, nframes);
	if(lines == NULL) {
		m_avail = false;
		return;
	}

	for(size_t i = 0; i < nframes; i++) {
		m_trace.push_back(std::string(lines[i]));
	}

	free(lines);
}
#else
backtrace::backtrace() : m_avail(false) {

}
#endif

} // namespace libtensor
