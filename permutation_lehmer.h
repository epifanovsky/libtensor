#ifndef __LIBTENSOR_PERMUTATION_LEHMER_H
#define __LIBTENSOR_PERMUTATION_LEHMER_H

#include <cstring>

#include "defs.h"
#include "exception.h"

namespace libtensor {

/**	\brief Implementation of permutations based on the Lehmer code
	\ingroup libtensor

	<h1>Lehmer code</h1>

	In an effort to represent permutations as single integers and not
	sequences of integers, the concept of the Lehmer code needs to be
	introduced first.

	Each permutation has a unique reduced representation that is based
	on the order of elementary transpositions that have to be applied to
	a sequence.

	Reference:
	A. Kerber, Applied Finite Group Actions, Springer-Verlag 1999

	<h1>Representation of Lehmer code by integers</h1>

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

	<h1>Remarks</h1>

	The Lehmer code doesn't contain the length of the sequence to which
	the permutation is to be applied. For example, a permutation of the
	two last items is coded the same way for any sequence that has more
	than two elements. Application of a permutation to a sequence that
	is shorter than expected, however, will cause an exception.
	So, it is the user's responsibility to keep track of the correctness
	of the dimensionality.
**/
class permutation_lehmer {
private:
	/**	\brief Permutation multiplication table
	**/
	class multable {
	private:
		
	public:
		multable(const size_t maxorder);
		size_t multiply(const size_t l, const size_t r) const;
	};

	/**	\brief Table of permutation inverse
	**/
	class invtable {
	};

private:
	static multable m_multable; //!< Permutation multiplication table
	static invtable m_invtable; //!< Permutation inverse table

	size_t m_code; //!< Lehmer code

public:
	/**	\brief Initializes the permutation to identity
		\param order Order of permutation (unused in this
			implementation)
	**/
	permutation_lehmer(const size_t order);

	/**	\brief Copy constructor
		\param p Another permutation
	**/
	permutation_lehmer(const permutation_lehmer &p);

	/**	\brief Returns \c true if this is the identity permutation,
			\c false otherwise
	**/
	bool is_identity() const;

	/**	\brief Returns \c true if two permutations are equal,
			\c false otherwise
	**/
	bool equals(const permutation_lehmer &p) const;

	/**	\brief Returns \c true if two permutations are equal,
			\c false otherwise
	**/
	bool operator==(const permutation_lehmer &p) const;

	/**	\brief Returns \c true if two permutations are different,
			\c false otherwise
	**/
	bool operator!=(const permutation_lehmer &p) const;

	/**	\brief Returns \c true if this permutation precedes another
			one in the lexicographical order, \c false otherwise
	**/
	bool operator<(const permutation_lehmer &p) const;

	/**	\brief Multiplies this permutation by another permutation
		\param p Another permutation
	**/
	void permute(const permutation_lehmer &p);

	/**	\brief Sets this permutation to the inverse of another one
		\param p Another permutation
	**/
	void inverse(const permutation_lehmer &p);

	/**	\brief Permutes a sequence of objects
		\param n Number of objects in sequence
		\param obj Array of objects
	**/
	template<class T>
	void apply(const size_t n, T *obj) const;

	/**	\brief Permutes a sequence of objects saving a separate output
		\param n Number of objects in sequence
		\param obj_from Array of objects
		\param[out] obj_to Array of permuted objects (output)
	**/
	template<class T>
	void apply(const size_t n, const T *obj_from, T *obj_to) const;
};

inline permutation_lehmer::permutation_lehmer(const size_t order) :
	m_code(0) {
}

inline permutation_lehmer::permutation_lehmer(const permutation_lehmer &p) :
	m_code(p.m_code) {
}

inline bool permutation_lehmer::is_identity() const {
	return m_code==0;
}

inline bool permutation_lehmer::equals(const permutation_lehmer &p) const {
	return m_code==p.m_code;
}

inline bool permutation_lehmer::operator==(const permutation_lehmer &p)
	const {
	return equals(p);
}

inline bool permutation_lehmer::operator!=(const permutation_lehmer &p)
	const {
	return !equals(p);
}

inline bool permutation_lehmer::operator<(const permutation_lehmer &p)
	const {
	return m_code<p.m_code;
}

inline void permutation_lehmer::permute(const permutation_lehmer &p) {
}

inline void permutation_lehmer::inverse(const permutation_lehmer &p) {
}

template<class T>
inline void permutation_lehmer::apply(const size_t n, T *obj) const {
}

template<class T>
inline void permutation_lehmer::apply(const size_t n, const T *obj_from,
	T *obj_to) const {
}

} // namespace libtensor

#endif // __LIBTENSOR_PERMUTATION_LEHMER_H

