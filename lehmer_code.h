#ifndef LIBTENSOR_LEHMER_CODE_H
#define LIBTENSOR_LEHMER_CODE_H

#include <vector>
#include <libvmm.h>

#include "defs.h"
#include "exception.h"
#include "permutation.h"

namespace libtensor {

/**	\brief Lehmer code for permutations: encoding and decoding

	Methods in this class convert permutations into single integers and
	back using a factorial representation of Lehmer code.

	<b>Lehmer code</b>

	In an effort to represent permutations as single integers and not
	sequences of integers, the concept of the Lehmer code needs to be
	introduced first.

	Each permutation has a unique reduced representation that is based
	on the order of elementary transpositions that have to be applied to
	a sequence.

	Reference:
	A. Kerber, Applied Finite Group Actions, Springer-Verlag 1999

	<b>Representation of Lehmer code by integers</b>

	The Lehmer code can be thought of in the framework of the factorial
	number system. Unlike the usual number system, in which a digit can
	run from 0 to n-1, where n is the base (usually 10). So, the
	denominations of the places come in powers of n. In the factorial
	system, the denominations of the places are the factorials of their
	order in the sequence: 1, 2, 6, 24, and so on. This system works
	because 1!+2!+3!+...+k!=(k+1)!-1.

	The numbers in the factorial system can be represented in the
	conventional system (binary in computers), and therefore single
	integers can be used to uniquely identify permutations.

	\ingroup libtensor
**/
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

