#ifndef __LIBTENSOR_PERMUTATION_H
#define __LIBTENSOR_PERMUTATION_H

#include <cstdio>

namespace libtensor {

/**
	\page permutations Permutations

	A %permutation is an object that contains information about how
	elements of a sequence need to be re-ordered. The most common use of
	permutations is in tensor classes and objects.


	\section introduction Introduction

	An implementation of %permutation (class) is used by the tensor class
	and like as a template parameter. The %permutation implementation is
	required to realize a given set of methods described in the 
	\ref interface Interface section. The implementation is also required
	to pass the test case described in the \ref tests Tests section.


	\section interface Interface

	For performance purposes, an implementation is not required to
	derive from a base abstract class (interface), but rather has to
	implement a given set of methods.

	\code
	class permutation_impl {
	public:
		// Constructors and destructor
		permutation_impl(const size_t order);
		permutation_impl(const permutation_impl &p);

		// Comparison methods and operators
		bool is_identity() const;
		bool equals(const permutation_impl &p) const;
		bool operator==(const permutation_impl &p) const;
		bool operator!=(const permutation_impl &p) const;
		bool operator<(const permutation_impl &p) const;

		// Operations
		void permute(const size_t i, const size_t j);
		void permute(const permutation_impl &p);
		void invert();
		void invert(const permutation_impl &p);

		template<class T>
		void apply(const size_t n, T *obj) const;
		template<class T>
		void apply(const size_t n, const T *obj_from, T *obj_to) const;
	};
	\endcode

	\subsection constdest Constructors and destructor

	Two constructors have to be implemented:

	\code
	permutation_impl(const size_t order);
	\endcode
	\c order is the order of the %permutation

	\code
	permutation_impl(const permutation_impl &p);
	\endcode
	Copy constructor. Creates an exact copy of another %permutation.

	A destructor should also be implemented if necessary.

	\subsection operators Comparison methods and operators

	The methods in this category do not throw exceptions. An appropriate
	return value should be used instead.

	\code
	bool is_identity() const;
	\endcode
	Returns \c true if the %permutation does not change the order in a
	sequence, \c false otherwise.

	\code
	bool equals(const permutation_impl &p) const;
	\endcode
	Returns \c true is the two %permutations are identical, that is they
	permute a sequence in the same way; \c false otherwise.

	\code
	bool operator==(const permutation_impl &p) const;
	\endcode
	Returns \c true is the two %permutations are identical, that is they
	permute a sequence in the same way; \c false otherwise. The same as
	\c equals().
	
	\code
	bool operator!=(const permutation_impl &p) const;
	\endcode
	Returns \c true if the two %permutations are different, \c false
	otherwise.

	\code
	bool operator<(const permutation_impl &p) const;
	\endcode
	Returns \c true if the %permutation precedes another %permutation in
	the lexicographical order, \c false otherwise.

	\subsection operations Operations

	The methods in this category can throw an exception, but if they
	do, they should be declared appropriately.

	\code
	void permute(const size_t i, const size_t j);
	\endcode
	Permutes two elements: \c i and \c j.
	
	\code
	void permute(const permutation_impl &p);
	\endcode
	Applies another %permutation on top of this %permutation.

	\code
	void invert();
	\endcode
	Inverts the %permutation.

	\code
	void invert(const permutation_impl &p);
	\endcode
	Assigns this %permutation as the inverse of another %permutation.

	\code
	template<class T>
	void apply(const size_t n, T *obj) const;
	\endcode
	Applies this %permutation to a sequence of objects \c obj of
	length \c n. If the re-arrangement is impossible for any reason,
	an exception is thrown.

	\code
	template<class T>
	void apply(const size_t n, const T *obj_from, T *obj_to) const;
	\endcode
	Applies this %permutation to a sequence of objects \c obj_from of
	length \c n and writes the permuted sequence to \c obj_to. If the
	re-arrangement is impossible for any reason, the method throws an
	exception.


	\section tests Tests


	\section implementations Implementations

	\li libtensor::permutation
	\li libtensor::permutation_lehmer

**/

/**	\brief Tensor %permutation
	\ingroup libtensor

	<b>Overview</b>

	A permutation object contains a permuted order or %tensor indices starting with 0.
	A unit %permutation doesn't change the order of indices, it is represented by a set
	of consecutive indices.

	For example, a fourth-order %tensor %index \e ijkl and a unit %permutation (0123).
	The %permutation elements are indices of the permuted entity in the unpermuted
	entity. The unit %permutation therefore doesn't change the order of indices.
	If we swap the first two indices, the %permutation becomes (1023), and the
	%tensor %index becomes \e jikl. If we then switch the first two pairs of indices,
	the %permutation becomes (2310), and the %index is \e klji.

	This example shows the above operations and prints out the resulting %index.
	\code
	char *idx = "ijkl"; // Starting index
	char perm_idx[5]; // Permuted index
	permutation p(4);

	// Operations on the permutation p
	p.permute(0,1);
	p.permute_pair(0,1);

	for(int i=0; i<4; i++) perm_idx[i] = idx[p[i]];
	perm_idx[4] = '\0';
	printf("%s -> %s", idx, perm_idx);
	\endcode

	<b>Inverse permutations</b>

	The inverse %permutation applied to the original %permutation gives a unit
	%permutation. The inverse of a unit %permutation is a unit %permutation.
	\f[ \mathcal{P} \mathcal{P}^{-1} = 1 \f]

	To obtain an inverse %permutation, the copy constructor or the invert() method
	can be used:
	\code
	permutation p(4);

	// Operations on the permutation p
	p.permute(0,1); // ijkl -> jikl
	p.permute_pair(0,1); // jikl -> klji

	permutation pc(p); // Plain copy
	permutation pi(p, true); // Inverse copy

	pc.invert(); // Now contains inverse permutation

	bool b_equals = pc.equals(pi); // true

	p.permute(pi); // p is now a unit permutation
	bool b_unit = p.is_unit(); // true
	\endcode

	<b>Permutables</b>

	Entities that can be permuted are called permutables an implement the interface
	tensor::permutable_i. The order of the permutable and the %permutation must be
	the same for the operation to be successful.
**/
class permutation {
private:
	unsigned int m_order; //!< Tensor order
	unsigned int m_idx[max_tensor_order]; //!< Permuted indices

	//!	\brief Exception class
	class permutation_exception : public exception {
	public:
		permutation_exception(const char *msg) : exception(msg) {}
		virtual ~permutation_exception() throw() {}
	};

public:
	/**	\brief Creates a unit %permutation of a specified order

		Creates a unit %permutation of a specified order.
		\param order Tensor order.
	**/
	permutation(const unsigned int order) throw(exception);

	/**	\brief Creates a copy or an inverted copy of a %permutation

		Creates a copy or an inverted copy of a %permutation.
		\param p Another %permutation.
		\param b_inverse Create inverse %permutation.
	**/
	permutation(const permutation &p, const bool b_inverse = false);

	/**	\brief Virtual destructor

		Virtual destructor.
	**/
	virtual ~permutation();

	/**	\brief Returns the order of the %permutation

		Returns the order of the %permutation.
	**/
	unsigned int get_order() const;

	/**	\brief Accumulates %permutation

		Permutes this %permutation with another %permutation, which must
		have the same order.

		\param p Another %permutation.
		\return Reference to this %permutation.
		\throw exception If the two permutations are incompatible.
	**/
	permutation &permute(const permutation &p) throw(exception);

	/**	\brief Permutes two indices

		Puts the first %index in the place of the second and the other
		way around. The %index values must not exceed the permutation order.

		\param i1 First %index.
		\param i2 Second %index.
		\return Reference to this %permutation.
		\throw exception If either of the indices is invalid.
	**/
	permutation &permute(const unsigned int i1, const unsigned int i2) throw(exception);

	/**	\brief Permutes two pairs of indices

		Puts the first %index pair in the place of the second and the other
		way around. For example, \e ijklmn after permuting pairs 0 and 1
		turns to \e klijmn.

		\param ip1 First %index pair.
		\param ip2 Second %index pair.
		\return Reference to this %permutation.
		\throw exception If either of the %index pairs is invalid.
	**/
	permutation &permute_pair(const unsigned int ip1, const unsigned int ip2)
		throw(exception);

	/**	\brief Inverts %permutation

		Inverts %permutation.

		\return Reference to this %permutation.
	**/
	permutation &invert();

	/**	\brief Checks if the %permutation is a unit %permutation

		Checks if the %permutation is a unit %permutation. A unit %permutation
		doesn't change the order of indices.

		\return true if this is a unit %permutation, false otherwise.
	**/
	bool is_unit() const;

	/**	\brief Checks if two permutations are identical

		Checks if two permutations are identical.

		\return true if the two permutations are equal, false otherwise.
	**/
	bool equals(const permutation &p) const;

	/**	\brief Returns a %permutation element

		Returns a %permutation element

		\return Index in the %permutation at the position \e i.
	**/
	unsigned int operator[](const unsigned int i) const throw(exception);

private:
	/**	\brief Throws an exception with an error message
	**/
	void throw_exc(const char *method, const char *msg) const throw(exception);
};

inline permutation::permutation(const unsigned int order) throw(exception) {
#ifdef TENSOR_DEBUG
	if(order == 0) throw_exc("permutation(const uint)", "Invalid permutation order");
#endif
	m_order = order;
	#pragma loop count(6)
	for(register unsigned int i=0; i<order; i++) m_idx[i] = i;
}

inline permutation::permutation(const permutation &p, const bool b_inverse) {
	m_order = p.m_order;
	if(b_inverse) {
		#pragma loop count(6)
		for(register unsigned int i=0; i<m_order; i++) m_idx[p.m_idx[i]] = i;
	} else {
		#pragma loop count(6)
		for(register unsigned int i=0; i<m_order; i++) m_idx[i] = p.m_idx[i];
	}
}

inline permutation::~permutation() {
}

inline unsigned int permutation::get_order() const {
	return m_order;
}

inline permutation &permutation::permute(const permutation &p) throw(exception) {
#ifdef TENSOR_DEBUG
	if(m_order != p.m_order) throw_exc("permute(const permutation&)",
		"Incompatible permutation");
#endif
	unsigned int idx_cp[max_tensor_order];
	#pragma loop count(6)
	for(register unsigned int i=0; i<m_order; i++) idx_cp[i] = m_idx[i];
	#pragma loop count(6)
	for(register unsigned int i=0; i<m_order; i++) m_idx[i] = idx_cp[p.m_idx[i]];
	return *this;
}

inline permutation &permutation::permute(const unsigned int i1, const unsigned int i2)
	throw(exception) {
#ifdef TENSOR_DEBUG
	if(i1 >= m_order || i2 >= m_order)
		throw_exc("permute(const uint, const uint)", "Index out of range");
#endif
	if(i1 == i2) return *this;
	register unsigned int i1_cp = m_idx[i1];
	m_idx[i1] = m_idx[i2]; m_idx[i2] = i1_cp;
	return *this;
}

inline permutation &permutation::permute_pair(const unsigned int ip1, const unsigned int ip2)
	throw(exception) {
	register unsigned int i1 = ip1*2, i2 = ip2*2;
#ifdef TENSOR_DEBUG
	if(i1+1 >= m_order || i2+1 >= m_order)
		throw_exc("permute_pair(const uint, const uint)", "Index out of range");
#endif
	if(i1 == i2) return *this;
	register unsigned int i1a_cp = m_idx[i1], i1b_cp = m_idx[i1+1];
	m_idx[i1] = m_idx[i2];
	m_idx[i1+1] = m_idx[i2+1];
	m_idx[i2] = i1a_cp;
	m_idx[i2+1] = i1b_cp;
	return *this;
}

inline permutation &permutation::invert() {
	unsigned int idx_cp[max_tensor_order];
	#pragma loop count(6)
	for(register unsigned int i=0; i<m_order; i++) idx_cp[i] = m_idx[i];
	#pragma loop count(6)
	for(register unsigned int i=0; i<m_order; i++) m_idx[idx_cp[i]] = i;
	return *this;
}

inline bool permutation::is_unit() const {
	#pragma loop count(6)
	for(register unsigned int i=0; i<m_order; i++) if(m_idx[i] != i) return false;
	return true;
}

inline bool permutation::equals(const permutation &p) const {
	if(m_order != p.m_order) return false;
	#pragma loop count(6)
	for(register unsigned int i=0; i<m_order; i++)
		if(m_idx[i] != p.m_idx[i]) return false;
	return true;
}

inline unsigned int permutation::operator[](const unsigned int i) const throw(exception) {
#ifdef TENSOR_DEBUG
	if(i >= m_order) throw_exc("operator[](cosnt uint)", "Index out of range");
#endif
	return m_idx[i];
}

inline void permutation::throw_exc(const char *method, const char *msg) const
	throw(exception) {
	char s[1024]; snprintf(s, 1024, "[tensor::permutation::%s] %s.", method, msg);
	throw permutation_exception(s);
}

} // namespace tensor

#endif // __TENSOR_PERMUTATION_H

