#ifndef LIBTENSOR_PERMUTATION_H
#define LIBTENSOR_PERMUTATION_H

#include <cstdio>
#include <iostream>

#include "defs.h"
#include "exception.h"

namespace libtensor {

/**	\brief Permutation of tensor indexes

	<b>Overview</b>

	A permutation object contains a permuted order or %tensor indices
	starting with 0. The identity %permutation doesn't change the order
	of the indexes.

	For example, a fourth-order %tensor %index \e ijkl and an identity
	%permutation (0123). The %permutation elements are indexes of the
	permuted entity in the unpermuted entity. The identity %permutation
	therefore doesn't change the order of the indexes. If we swap the first
	two indices, the %permutation becomes (1023), and the %tensor %index
	becomes \e jikl. If we then switch the first two pairs of indices, the
	%permutation becomes (2310), and the %index is \e klji.

	This example shows the above operations and prints out the resulting
	%index.
	\code
	char idx[5], perm_idx[5];
	strcpy(idx, "ijkl");
	strcpy(perm_idx, "ijkl");
	permutation p(4);

	// Operations on the permutation p
	p.permute(0,1); // jikl
	p.permute(0,2).permute(1,3); // klji

	p.apply(4, idx, perm_idx);
	printf("%s -> %s", idx, perm_idx); // prints ijkl -> klji
	\endcode

	<b>Inverse permutations</b>

	The inverse %permutation applied to the original %permutation yields
	the identity %permutation. The inverse of the identity %permutation or
	an elementary %permutation (%permutation of just two indexes) is the
	%permutation itself.
	\f[ \mathcal{P} \mathcal{P}^{-1} = 1 \f]

	To obtain the inverse %permutation, the copy constructor or the
	invert() method can be used:
	\code
	permutation p(4);

	// Operations on the permutation p
	p.permute(0,1); // ijkl -> jikl
	p.permute(0,2).permute(1,3); // jikl -> klji

	permutation pc(p); // Plain copy
	permutation pi(p, true); // Inverse copy

	pc.invert(); // Now contains inverse permutation

	bool b_equals = pc.equals(pi); // true

	p.permute(pi); // p is now a unit permutation
	bool b_unit = p.is_identity(); // true
	\endcode

	\ingroup libtensor_core
**/
template<size_t N>
class permutation {
private:
	size_t m_idx[N]; //!< Permuted indices

public:
	/**	\brief Creates a unit %permutation of a specified order
	**/
	permutation();

	/**	\brief Creates a copy or an inverted copy of a %permutation

		Creates a copy or an inverted copy of a %permutation.
		\param p Another %permutation.
		\param b_inverse Create inverse %permutation.
	**/
	permutation(const permutation<N> &p, const bool b_inverse = false);

	/**	\brief Accumulates %permutation

		Permutes this %permutation with another %permutation.

		\param p Another %permutation.
		\return Reference to this %permutation.
	**/
	permutation<N> &permute(const permutation<N> &p);

	/**	\brief Permutes two indexes
		\param i First %index.
		\param j Second %index.
		\return Reference to this %permutation.
		\throw exception If either of the indices is invalid (out of
			range).
	**/
	permutation<N> &permute(const size_t i, const size_t j)
		throw(exception);

	/**	\brief Inverts %permutation
		\return Reference to this %permutation.
	**/
	permutation<N> &invert();

	/**	\brief Resets %permutation (makes it an identity %permutation)
	 **/
	void reset();

	/**	\brief Checks if the %permutation is an identity %permutation

		Checks if the %permutation is an identity %permutation.
		An identity %permutation doesn't change the order in
		a sequence.

		\return true if this is an identity %permutation, false
		otherwise.
	**/
	bool is_identity() const;

	/**	\brief Checks if two permutations are identical
		\return true if the two permutations are equal, false otherwise.
	**/
	bool equals(const permutation<N> &p) const;

	bool operator==(const permutation<N> &p) const;
	bool operator!=(const permutation<N> &p) const;
	bool operator<(const permutation<N> &p) const;

	/**	\brief Permutes a given sequence of objects
		\param n Length of the sequence, must be the same as the
			permutation order
		\param obj Pointer to the sequence
	**/
	template<class T>
	void apply(const size_t n, T *obj) const throw(exception);

	template<typename T>
	void apply(T (&seq)[N]) const;

	/**	\brief Permutes a given sequence of objects and writes the
			result to a different location
		\param n Length of the sequence, must be the same as the
			permutation order
		\param obj_from Pointer to the initial sequence
		\param obj_to Pointer to the resulting sequence
	**/
	template<class T>
	void apply(const size_t n, const T *obj_from, T *obj_to) const
		throw(exception);

};

template<size_t N>
inline permutation<N>::permutation() {
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) m_idx[i] = i;
}

template<size_t N>
inline permutation<N>::permutation(const permutation<N> &p,
	const bool b_inverse) {
	if(b_inverse) {
		#pragma unroll(N)
		for(register size_t i=0; i<N; i++) m_idx[p.m_idx[i]] = i;
	} else {
		#pragma unroll(N)
		for(register size_t i=0; i<N; i++) m_idx[i] = p.m_idx[i];
	}
}

template<size_t N>
inline permutation<N> &permutation<N>::permute(const permutation<N> &p) {
	size_t idx_cp[N];
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) idx_cp[i] = m_idx[i];
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) m_idx[i] = idx_cp[p.m_idx[i]];
	return *this;
}

template<size_t N>
inline permutation<N> &permutation<N>::permute(const size_t i, const size_t j)
	throw(exception) {
#ifdef LIBTENSOR_DEBUG
	if(i>=N || j>=N) {
		throw_exc("permutation<N>",
			"permute(const size_t, const size_t)",
			"Index out of range");
	}
#endif // LIBTENSOR_DEBUG
	if(i == j) return *this;
	register size_t i_cp = m_idx[i];
	m_idx[i] = m_idx[j]; m_idx[j] = i_cp;
	return *this;
}

template<size_t N>
inline permutation<N> &permutation<N>::invert() {
	size_t idx_cp[N];
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) idx_cp[i] = m_idx[i];
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) m_idx[idx_cp[i]] = i;
	return *this;
}

template<size_t N>
inline void permutation<N>::reset() {
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) m_idx[i] = i;
}

template<size_t N>
inline bool permutation<N>::is_identity() const {
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++)
		if(m_idx[i] != i) return false;
	return true;
}

template<size_t N>
inline bool permutation<N>::equals(const permutation<N> &p) const {
	if(&p == this) return true;
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++)
		if(m_idx[i] != p.m_idx[i]) return false;
	return true;
}

template<size_t N>
inline bool permutation<N>::operator==(const permutation<N> &p) const {
	return equals(p);
}

template<size_t N>
inline bool permutation<N>::operator!=(const permutation<N> &p) const {
	return !equals(p);
}

template<size_t N>
inline bool permutation<N>::operator<(const permutation<N> &p) const {
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) {
		if(m_idx[i] != p.m_idx[i]) return m_idx[i]<p.m_idx[i];
	}
	return false;
}

template<size_t N> template<class T>
void permutation<N>::apply(const size_t n, T *obj) const throw(exception) {
#ifdef LIBTENSOR_DEBUG
	if(n!=N) {
		throw_exc("permutation<N>", "apply(const size_t, T*)",
			"Sequence has a wrong length");
	}
#endif // LIBTENSOR_DEBUG
	T buf[N];
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) buf[i]=obj[i];
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) obj[i]=buf[m_idx[i]];
}

template<size_t N> template<typename T>
void permutation<N>::apply(T (&seq)[N]) const {
	T buf[N];
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) buf[i]=seq[i];
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) seq[i]=buf[m_idx[i]];
}

template<size_t N> template<class T>
void permutation<N>::apply(const size_t n, const T *obj_from, T *obj_to) const
	throw(exception) {
#ifdef LIBTENSOR_DEBUG
	if(n!=N) {
		throw_exc("permutation<N>", "apply(const size_t, const T*, T*)",
			"Sequence has a wrong length");
	}
#endif // LIBTENSOR_DEBUG
	#pragma unroll(N)
	for(register size_t i=0; i<N; i++) obj_to[i] = obj_from[m_idx[i]];
}

/**	\brief Prints out a permutation to an output stream

	\ingroup libtensor
**/
template<size_t N>
std::ostream &operator<<(std::ostream &os, const permutation<N> &p) {
	static const char *alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	char seq1[N+1], seq2[N+1];
	for(size_t i=0; i<N; i++) seq1[i]=seq2[i]=alphabet[i];
	seq1[N]=seq2[N]='\0';
	p.apply(N, seq1, seq2);
	os << "[" << seq1 << "->" << seq2 << "]";
	return os;
}

} // namespace libtensor

#endif // LIBTENSOR_PERMUTATION_H

