#ifndef LIBTENSOR_LEHMER_CODE_H
#define LIBTENSOR_LEHMER_CODE_H

#include <vector>
#include <libvmm.h>

#include "defs.h"
#include "exception.h"
#include "permutation.h"

namespace libtensor {

class lehmer_code : public libvmm::singleton<lehmer_code> {
	friend class libvmm::singleton<lehmer_code>;

private:
	size_t m_fact[max_tensor_order-1]; //!< Table of factorials

	//!	Table of permutations
	std::vector<permutation*> m_codes[max_tensor_order-1];

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
	const permutation &code2perm(const size_t order, const size_t code)
		throw(exception);

	//@}

private:
	/**	\brief Throws an exception
	**/
	void throw_exc(const char *method, const char *msg) const
		throw(exception);
};

} // namespace libtensor

#endif // LIBTENSOR_LEHMER_CODE_H

