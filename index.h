#ifndef __TENSOR_INDEX_H
#define __TENSOR_INDEX_H

#include "defs.h"
#include "exception.h"

namespace tensor {

/**	\brief Indexes a %tensor element
	\ingroup tensor
	\author Evgeny Epifanovsky

	<b>Permutations</b>

	Elements of the %index can be permuted (class implements tensor::permutable_i).

	<b>Exceptions</b>

	Any %exception thrown leaves the index object unchanged.
**/
class index {
private:
	unsigned int m_order; //!< Tensor order
	size_t m_idx[max_tensor_order]; //!< Tensor %index

	//!	\brief Exception class
	class index_exception : public exception {
	public:
		index_exception(const char *msg) : exception(msg) {}
		virtual ~index_exception() throw() {}
	};

public:
	/**	\brief Creates the %index of the first element of a %tensor with a given order
		\param order Tensor order.
	**/
	index(const unsigned int order);

	/**	\brief Copies the %index from another instance
		\param idx Another %index.
	**/
	index(const index &idx);

	/**	\brief Virtual destructor
	**/
	virtual ~index();

	/**	\brief Returns the order of the %index

		Returns the order of the %index.
	**/
	unsigned int get_order() const;

	/**	\brief Checks if two indices are equal

		Returns true if the indices are equal, false otherwise.
	**/
	bool equals(const index &idx) const;

	/**	\brief Lvalue individual element accessor

		Returns an individual %index element at the position \e i as an
		lvalue.

		\param i Element position.
		\return Reference to the element.
		\throw exception If the position is out of range.
	**/
	size_t &operator[](const unsigned int i) throw(exception);

	/**	\brief Rvalue individual element accessor

		Returns an individual %index element at the position \e i as an
		rvalue.

		\param i Element position.
		\return Element value.
		\throw exception If the position is out of range.
	**/
	size_t operator[](const unsigned int i) const throw(exception);

	//!	\name Implementation of tensor::permutable_i
	//@{
	/**	\brief Permutes the elements of the %index
		\param p Permutation.
		\return Reference to this %index.
		\throw exception If the %index and the %permutation are incompatible.
	**/
	virtual index &permute(const permutation &p) throw(exception);
	//@}

private:
	/**	\brief Throws an exception
	**/
	void throw_exc(const char *method, const char *msg) const throw(exception);
};

inline index::index(const unsigned int order) {
#ifdef TENSOR_DEBUG
	if(order == 0 || order >= max_tensor_order)
		throw_exc("index(const uint)", "Index order is not allowed");
#endif
	m_order = order;
	#pragma loop count(6)
	for(register unsigned int i=0; i<m_order; i++) m_idx[i] = 0;
}

inline index::index(const index &idx) : m_order(idx.m_order) {
	#pragma loop count(6)
	for(register unsigned int i=0; i<m_order; i++) m_idx[i] = idx.m_idx[i];
}

inline index::~index() {
}

inline unsigned int index::get_order() const {
	return m_order;
}

inline bool index::equals(const index &idx) const {
	if(m_order != idx.m_order) return false;
	#pragma loop count(6)
	for(register unsigned int i=0; i<m_order; i++)
		if(m_idx[i] != idx.m_idx[i]) return false;
	return true;
}

inline size_t &index::operator[](const unsigned int i) throw(exception) {
#ifdef TENSOR_DEBUG
	if(i >= m_order) throw_exc("operator[](const uint)",
		"Index out of range");
#endif
	return m_idx[i];
}

inline size_t index::operator[](const unsigned int i) const throw(exception) {
#ifdef TENSOR_DEBUG
	if(i >= m_order) throw_exc("operator[](const uint) const",
		"Index out of range");
#endif
	return m_idx[i];
}

inline index &index::permute(const permutation &p) throw(exception) {
#ifdef TENSOR_DEBUG
	if(m_order != p.get_order()) throw_exc("permute(const permutation&)",
		"Incompatible permutation");
#endif
	size_t idx_cp[max_tensor_order];
	#pragma loop count(6)
	for(register unsigned int i=0; i<m_order; i++) idx_cp[i] = m_idx[i];
	#pragma loop count(6)
	for(register unsigned int i=0; i<m_order; i++) m_idx[i] = idx_cp[p[i]];
	return *this;
}

inline void index::throw_exc(const char *method, const char *msg) const
	throw(exception) {
	char s[1024]; snprintf(s, 1024, "[tensor::index::%s] %s.", method, msg);
	throw index_exception(s);
}

} // namespace tensor

#endif // __TENSOR_INDEX_H

