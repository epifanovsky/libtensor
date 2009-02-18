#ifndef __LIBTENSOR_INDEX_H
#define __LIBTENSOR_INDEX_H

#include "defs.h"
#include "exception.h"

namespace libtensor {

/**	\brief Index of a single %tensor element
	\ingroup libtensor

	A correct %index must have the same order as the %tensor, and none of
	the %index elements must be out of the range of the %tensor
	%dimensions.

	The elements of an %index can be permuted by a %permutation. Since
	there can be multiple implementations of permutations, the method
	permute() is a template. For more info \ref permutations.
**/
class index {
private:
	size_t m_order; //!< Tensor order
	size_t m_idx[max_tensor_order]; //!< Tensor %index

public:
	/**	\brief Creates the %index of the first element of a %tensor
			 with a given order
		\param order Tensor order.
	**/
	index(const size_t order);

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
	size_t get_order() const;

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

	/**	\brief Permutes the elements of the %index
		\param p Permutation.
		\return Reference to this %index.
		\throw exception If the %index and the %permutation are
			incompatible.
	**/
	template<class Perm>
	index &permute(const Perm &p) throw(exception);

private:
	/**	\brief Throws an exception
	**/
	void throw_exc(const char *method, const char *msg) const throw(exception);
};

inline index::index(const size_t order) {
#ifdef TENSOR_DEBUG
	if(order == 0 || order >= max_tensor_order)
		throw_exc("index(size_t)", "Index order is not allowed");
#endif
	m_order = order;
	#pragma loop count(6)
	for(register size_t i=0; i<m_order; i++) m_idx[i] = 0;
}

inline index::index(const index &idx) : m_order(idx.m_order) {
	#pragma loop count(6)
	for(register unsigned int i=0; i<m_order; i++) m_idx[i] = idx.m_idx[i];
}

inline index::~index() {
}

inline size_t index::get_order() const {
	return m_order;
}

inline bool index::equals(const index &idx) const {
	if(m_order != idx.m_order) return false;
	#pragma loop count(6)
	for(register unsigned int i=0; i<m_order; i++)
		if(m_idx[i] != idx.m_idx[i]) return false;
	return true;
}

inline size_t &index::operator[](const size_t i) throw(exception) {
#ifdef TENSOR_DEBUG
	if(i >= m_order)
		throw_exc("operator[](const size_t)", "Index out of range");
#endif
	return m_idx[i];
}

inline size_t index::operator[](const size_t i) const throw(exception) {
#ifdef TENSOR_DEBUG
	if(i >= m_order) {
		throw_exc("operator[](const size_t) const",
			"Index out of range");
	}
#endif
	return m_idx[i];
}

template<class Perm>
inline index &index::permute(const Perm &p) throw(exception) {
#ifdef TENSOR_DEBUG
	if(m_order != p.get_order())
		throw_exc("permute(const Perm&)", "Incompatible permutation");
#endif
	p.apply(m_order, m_idx);
	return *this;
}

inline void index::throw_exc(const char *method, const char *msg) const
	throw(exception) {
	char s[1024];
	snprintf(s, 1024, "[libtensor::index::%s] %s.", method, msg);
	throw exception(s);
}

} // namespace libtensor

#endif // __LIBTENSOR_INDEX_H

