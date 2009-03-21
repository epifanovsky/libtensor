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
template<size_t N>
class lehmer_code : public libvmm::singleton< lehmer_code<N> > {
	friend class libvmm::singleton< lehmer_code<N> >;

private:
	size_t m_fact[N]; //!< Table of factorials
	permutation<N> **m_codes; //!< Table of permutations

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
	size_t perm2code(const permutation<N> &p) throw(exception);

	/**	\brief Returns the %permutation for a Lehmer code
	**/
	const permutation<N> &code2perm(const size_t code) throw(exception);

	//@}

};

template<size_t N>
lehmer_code<N>::lehmer_code() {
	// m_fact[0] = 1!
	// m_fact[1] = 2!
	// m_fact[2] = 3!
	// ...
	// m_fact[n] = (n+1)!
	size_t fact = 1;
	for(register size_t i=0; i<N; i++) {
		fact *= (i+1); m_fact[i] = fact;
	}
	register size_t sz = m_fact[N-1];
	m_codes = new permutation<N>*[sz];
	for(register size_t i=0; i<sz; i++) m_codes[i] = NULL;
}

template<size_t N>
size_t lehmer_code<N>::perm2code(const permutation<N> &p) throw(exception) {
	size_t seq[N];
	size_t code = 0;
	for(register size_t i=0; i<N; i++) seq[i]=i;
	p.apply(N, seq);
	for(size_t i=0; i<N-1; i++) {
		register size_t k = seq[i];
		for(register size_t j=i+1; j<N; j++) if(seq[j]>k) seq[j]--;
		code += k*m_fact[N-i-2];
	}
	return code;
}

template<size_t N>
const permutation<N> &lehmer_code<N>::code2perm(const size_t code)
	throw(exception) {
#ifdef LIBTENSOR_DEBUG
	if(code >= m_fact[N-1]) {
		throw_exc("lehmer_code<N>", "code2perm(const size_t)",
			"Invalid code");
	}
#endif // LIBTENSOR_DEBUG
	permutation<N> *p = m_codes[code];
	if(p) return *p;

	p = new permutation<N>;

	size_t c = code;
	size_t seq[N-1];
	register size_t i = N-1;
	do {
		i--;
		seq[i] = c/m_fact[i];
		c = c%m_fact[i];
	} while(i != 0);

	bool done = false;
	do {
		i = 0;
		while(i<N-1 && seq[i]==0) i++;
		if(i!=N-1) {
			p->permute(N-i-2, N-i-1);
			if(i==0) seq[i]=0;
			else { seq[i-1]=seq[i]-1; seq[i]=0; }
		} else {
			done = true;
		}
	} while(!done);
	p->invert();

	m_codes[code] = p;
	return *p;
}

} // namespace libtensor

#endif // LIBTENSOR_LEHMER_CODE_H

