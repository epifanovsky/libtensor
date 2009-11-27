#ifndef LIBTENSOR_VERSION_H
#define LIBTENSOR_VERSION_H

#include "defs.h"
#include "exception.h"

namespace libtensor {


/**	\brief Version of the %tensor library

	The %version of the library is specified using the %version number and
	status. The number consists of a major number and a minor number. The
	status string describes the release status.

	For example, %version 2.0-alpha2 has major number 2, minor number 0,
	and status "alpha2" meaning the second alpha release.

	\ingroup libtensor
 **/
class version {
private:
	//!	Major %version number
	static const unsigned m_major = 2;

	//!	Minor %version number
	static const unsigned m_minor = 0;

	//!	Version status
	static const char* const m_status;//= "alpha2";

	//!	Version string
	static const char* const m_string;//= "2.0-alpha2";

public:
	/**	\brief Returns the major %version number
	 **/
	static unsigned get_major() {
		return m_major;
	}

	/**	\brief Returns the minor %version number
	 **/
	static unsigned get_minor() {
		return m_minor;
	}

	/**	\brief Returns the %version status
	 **/
	static const char *get_status() {
		return m_status;
	}

	/**	\brief Returns the string that corresponds to the %version
	 **/
	static const char *get_string() {
		return m_string;
	}
};

} // namespace libtensor

#endif // LIBTENSOR_VERSION_H
