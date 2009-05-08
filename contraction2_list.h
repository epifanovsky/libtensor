#ifndef LIBTENSOR_CONTRACTION2_LIST_H
#define	LIBTENSOR_CONTRACTION2_LIST_H

#include "defs.h"
#include "exception.h"

namespace libtensor {

/**	\brief Contraction loop node

	Stores the loops over which a contraction should be done and offers
	a few very specific methods to operate on them.

	\ingroup libtensor
 **/
struct contraction2_node {
	size_t m_weight;
	size_t m_inca;
	size_t m_incb;
	size_t m_incc;
};

/**	\brief List of contraction loop nodes
	\tparam N Maximum number of nodes

	\ingroup libtensor
 **/
template<size_t N>
class contraction2_list {
private:
	static const size_t k_invalid = (size_t) (-1);
	contraction2_node m_nodes[N];
	size_t m_len; //!< Number of nodes
	size_t m_first; //!< First node
	size_t m_last; //!< Last node
	size_t m_next[N]; //!< References to next node
	size_t m_prev[N]; //!< References to previous node

public:

	/**	\brief Default constructor
	 **/
	contraction2_list();

	/**	\brief Returns the length of the list
	 **/
	size_t length() const;

	/**	\brief Returns the first node
	 **/
	size_t first() const;

	/**	\brief Returns the last node
	 **/
	size_t last() const;

	/**	\brief Returns next node
	 **/
	size_t next(size_t node) const throw(exception);

	/**	\brief Returns previous node
	 **/
	size_t prev(size_t node) const throw(exception);

	/**	\brief Appends a node to the end of the list
		\param weight Node weight.
		\param inca Increment of the first argument.
		\param incb Increment of the second argument.
		\param incc Increment of the result.
	 **/
	void append(size_t weight, size_t inca, size_t incb, size_t incc)
	throw(exception);
};

template<size_t N>
contraction2_list::contraction2_list() : m_len(0), m_first(k_invalid),
m_last(k_invalid) {
	for(size_t i = 0; i < N; i++) {
		m_next[i] = k_invalid;
		m_prev[i] = k_invalid;
	}
}

template<size_t N>
inline size_t contraction2_list::length() const {
	return m_len;
}

template<size_t N>
inline size_t contraction2_list::first() const {
	return m_first;
}

template<size_t N>
inline size_t contraction2_list::last() const {
	return m_last;
}

template<size_t N>
inline size_t contraction2_list::next(size_t node) const {
	if(node == k_invalid || node >= N) {
		throw_exc("contraction2_list", "next(size_t)",
			"Invalid node number");
	}
	return m_next[node];
}

template<size_t N>
inline size_t contraction2_list::prev(size_t node) const {
	if(node == k_invalid || node >= N) {
		throw_exc("contraction2_list", "prev(size_t)",
			"Invalid node number");
	}
	return m_prev[node];
}

template<size_t N>
void contraction2_list::append(size_t weight, size_t inca, size_t incb,
	size_t incc) throw(exception) {

	if(m_len == N)
		throw_exc("contraction2_list", "append()", "List is full");

	m_nodes[m_len].m_weight = weight;
	m_nodes[m_len].m_inca = inca;
	m_nodes[m_len].m_incb = incb;
	m_nodes[m_len].m_incc = incc;
	if(m_len == 0) {
		m_first = m_len;
	} else {
		m_next[m_last] = m_len;
		m_prev[m_len] = m_last;
	}
	m_last = m_len;
	m_len++;
}

} // namespace libtensor

#endif // LIBTENSOR_CONTRACTION2_LIST_H

