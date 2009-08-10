#ifndef LIBTENSOR_SEQUENCE_H
#define LIBTENSOR_SEQUENCE_H

#include "defs.h"
#include "exception.h"
#include "permutation.h"

namespace libtensor {

/**	\brief Sequence of objects of given type and length
	\tparam N Sequence length.
	\tparam T Object type.

	\ingroup libtensor_core
 **/
template<size_t N, typename T>
class sequence {
public:
	static const char *k_clazz; //!< Class name

private:
	T m_seq[N]; //!< Sequence

public:
	//!	\name Construction and destruction
	//@{

	sequence(const T t);
	sequence(const sequence<N, T> &seq);

	//@}

	//!	\name Access to %sequence elements, manipulations
	//@{

	/**	\brief Returns the reference to an element at a given position
			(l-value)
		\param pos Position (not to exceed N).
		\throw out_of_bounds If the position exceeds N.
	 **/
	T &at(size_t pos) throw(out_of_bounds);

	/**	\brief Returns an element at a given position (r-value)
		\param pos Position (not to exceed N).
		\throw out_of_bounds If the position exceeds N.
	 **/
	T at(size_t pos) const throw(out_of_bounds);

	/**	\brief Permutes the sequence
		\param p Permutation.
	 **/
	void permute(const permutation<N> &perm);

	//@}

	//!	\name Overloaded operators
	//@{

	/**	\brief Returns the reference to an element at a given position
			(l-value)
		\param pos Position (not to exceed N, %index order).
		\throw out_of_bounds If the position exceeds N.
	 **/
	T &operator[](size_t pos) throw(out_of_bounds);

	/**	\brief Returns an element at a given position (r-value)
		\param pos Position (not to exceed N, %index order).
		\throw out_of_bounds If the position exceeds N.
	 **/
	T operator[](size_t pos) const throw(out_of_bounds);

	//@}

protected:
	//!	\name Protected access to %sequence elements
	//@{

	/**	\brief Returns the reference to an element at a given position
			(l-value); no checks done
		\param pos Position (not to exceed N).
	 **/
	T &at_nochk(size_t pos);

	/**	\brief Returns an element at a given position (r-value); no
			checks done
		\param pos Position (not to exceed N).
	 **/
	T at_nochk(size_t pos) const;

	//@}

};


template<size_t N, typename T>
const char *sequence<N, T>::k_clazz = "sequence<N, T>";


template<size_t N, typename T>
inline sequence<N, T>::sequence(const T t) {

	for(register size_t i = 0; i < N; i++) m_seq[i] = t;
}


template<size_t N, typename T>
inline sequence<N, T>::sequence(const sequence<N, T> &seq) {

	for(register size_t i = 0; i < N; i++) m_seq[i] = seq.m_seq[i];
}


template<size_t N, typename T>
inline T &sequence<N, T>::at(size_t pos) throw(out_of_bounds) {

#ifdef LIBTENSOR_DEBUG
	if(pos >= N) {
		throw out_of_bounds("libtensor", k_clazz, "at(size_t)",
			__FILE__, __LINE__, "pos");
	}
#endif // LIBTENSOR_DEBUG
	return at_nochk(pos);
}


template<size_t N, typename T>
inline T sequence<N, T>::at(size_t pos) const throw(out_of_bounds) {

#ifdef LIBTENSOR_DEBUG
	if(pos >= N) {
		throw out_of_bounds("libtensor", k_clazz, "at(size_t) const",
			__FILE__, __LINE__, "pos");
	}
#endif // LIBTENSOR_DEBUG
	return at_nochk(pos);
}


template<size_t N, typename T>
inline void sequence<N, T>::permute(const permutation<N> &perm) {

	perm.apply(m_seq);
}


template<size_t N, typename T>
inline T &sequence<N, T>::operator[](size_t pos) throw(out_of_bounds) {

	return at(pos);
}


template<size_t N, typename T>
inline T sequence<N, T>::operator[](size_t pos) const throw(out_of_bounds) {

	return at(pos);
}


template<size_t N, typename T>
inline T &sequence<N, T>::at_nochk(size_t pos) {

	return m_seq[pos];
}


template<size_t N, typename T>
inline T sequence<N, T>::at_nochk(size_t pos) const {

	return m_seq[pos];
}


} // namespace libtensor

#endif // LIBTENSOR_SEQUENCE_H
