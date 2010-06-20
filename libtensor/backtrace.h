#ifndef LIBTENSOR_BACKTRACE_H
#define LIBTENSOR_BACKTRACE_H

#include <list>
#include <string>
#include "defs.h"

namespace libtensor {

class backtrace {
public:
	typedef std::list<std::string>::const_iterator iterator;

private:
	bool m_avail;
	std::list<std::string> m_trace;

public:
	backtrace();

	bool is_avail() const {
		return m_avail;
	}
	iterator begin() const {
		return m_trace.begin();
	}
	iterator end() const {
		return m_trace.end();
	}

};

} // namespace libtensor

#endif // LIBTENSOR_BACKTRACE_H
