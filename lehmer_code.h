#ifndef LIBTENSOR_LEHMER_CODE_H
#define LIBTENSOR_LEHMER_CODE_H

#include <libvmm.h>

#include "defs.h"
#include "exception.h"
#include "permutation.h"

namespace libtensor {

class lehmer_code : public libvmm::singleton<lehmer_code> {
	friend libvmm::singleton<lehmer_code>;

protected:
	//!	\name Construction and destruction
	//@{

	/**	\brief Protected singleton constructor
	**/
	lehmer_code();

	//@}

public:
	//!	\name Lehmer encoding and decoding
	//@{

	/**	\brief Returns the Lehmer code for a %permutation
	**/
	size_t perm2code(const permutation &p) throw(exception);

	/**	\brief Returns the %permutation for a Lehmer code
	**/
	const permutation &code2perm(const size_t code) throw(exception);

	//@}
};

} // namespace libtensor

#endif // LIBTENSOR_LEHMER_CODE_H

