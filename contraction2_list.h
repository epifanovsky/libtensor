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
	size_t get_length() const;

	/**	\brief Returns the first node number
	 **/
	size_t get_first() const;

	/**	\brief Returns the last node number
	 **/
	size_t get_last() const;

	/**	\brief Returns next node number
	 **/
	size_t get_next(size_t node) const throw(exception);

	/**	\brief Returns previous node number
	 **/
	size_t get_prev(size_t node) const throw(exception);

	/**	\brief Returns the reference to a node
	 **/
	const contraction2_node &get_node(size_t node) const throw(exception);

	/**	\brief Appends a node to the end of the list
		\param weight Node weight.
		\param inca Increment of the first argument.
		\param incb Increment of the second argument.
		\param incc Increment of the result.
	 **/
	void append(size_t weight, size_t inca, size_t incb, size_t incc)
	throw(exception);

private:
	void detach(size_t node);
	void attach_end(size_t node);
};

template<size_t N>
contraction2_list<N>::contraction2_list() : m_len(0), m_first(k_invalid),
m_last(k_invalid) {
	for(size_t i = 0; i < N; i++) {
		m_next[i] = k_invalid;
		m_prev[i] = k_invalid;
	}
}

template<size_t N>
inline size_t contraction2_list<N>::get_length() const {
	return m_len;
}

template<size_t N>
inline size_t contraction2_list<N>::get_first() const {
	return m_first;
}

template<size_t N>
inline size_t contraction2_list<N>::get_last() const {
	return m_last;
}

template<size_t N>
inline size_t contraction2_list<N>::get_next(size_t node) const throw(exception) {
	if(node == k_invalid || node >= N) {
		throw_exc("contraction2_list", "get_next(size_t)",
			"Invalid node number");
	}
	return m_next[node];
}

template<size_t N>
inline size_t contraction2_list<N>::get_prev(size_t node) const
throw(exception) {
	if(node == k_invalid || node >= N) {
		throw_exc("contraction2_list", "get_prev(size_t)",
			"Invalid node number");
	}
	return m_prev[node];
}

template<size_t N>
inline const contraction2_node &contraction2_list<N>::get_node(size_t node)
const throw(exception) {
	if(node == k_invalid || node >= N) {
		throw_exc("contraction2_list", "get_node(size_t)",
			"Invalid node number");
	}
	return m_nodes[node];
}

template<size_t N>
void contraction2_list<N>::append(size_t weight, size_t inca, size_t incb,
	size_t incc) throw(exception) {

	if(m_len == N)
		throw_exc("contraction2_list", "append()", "List is full");

	m_nodes[m_len].m_weight = weight;
	m_nodes[m_len].m_inca = inca;
	m_nodes[m_len].m_incb = incb;
	m_nodes[m_len].m_incc = incc;
	attach_end(m_len);
}

template<size_t N>
void contraction2_list<N>::detach(size_t node) {
}

template<size_t N>
void contraction2_list<N>::attach_end(size_t node) {
	if(m_len == 0) {
		m_first = node;
		m_next[node] = k_invalid;
		m_prev[node] = k_invalid;
	} else {
		m_next[m_last] = node;
		m_prev[node] = m_last;
		m_next[node] = k_invalid;
	}
	m_last = node;
	m_len++;
}

} // namespace libtensor

#endif // LIBTENSOR_CONTRACTION2_LIST_H

