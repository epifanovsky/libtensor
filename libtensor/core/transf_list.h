#ifndef LIBTENSOR_TRANSF_LIST_H
#define LIBTENSOR_TRANSF_LIST_H

#include <algorithm>
#include <list>
#include "abs_index.h"
#include "symmetry.h"
#include "transf.h"

namespace libtensor {


/**	\brief Enumerates all transformations associated with a block in
		a %symmetry group
	\tparam N Tensor order (symmetry cardinality).
	\tparam T Tensor element type.

	This algorithm applies all elements in a given symmetry group
	until it exhausts all the transformations associated with a given
	block.

	\ingroup libtensor_core
 **/
template<size_t N, typename T>
class transf_list {
public:
	static const char *k_clazz; //!< Class name

public:
	typedef typename std::list< transf<N, T> >::const_iterator
		iterator; //!< List iterator

private:
	std::list< transf<N, T> > m_trlist;

public:
	/**	\brief Constructs the list of transformations
		\param sym Symmetry group.
		\param idx Block %index.
	 **/
	transf_list(const symmetry<N, T> &sym, const index<N> &idx);

	/**	\brief Returns true if the transformation is listed,
			false otherwise
	 **/
	bool is_found(const transf<N, T> &tr) const;

	//!	\name STL-like list iterator
	//@{

	iterator begin() const {

		return m_trlist.begin();
	}

	iterator end() const {

		return m_trlist.end();
	}

	const transf<N, T> &get_transf(iterator &i) const {

		return *i;
	}

	//@}

private:
	void make_list(const symmetry<N, T> &sym, const index<N> &idx,
		const transf<N, T> &tr);

};


template<size_t N, typename T>
const char *transf_list<N, T>::k_clazz = "transf_list<N, T>";


template<size_t N, typename T>
transf_list<N, T>::transf_list(const symmetry<N, T> &sym, const index<N> &idx) {

	transf<N, T> tr0;
	make_list(sym, idx, tr0);
}


template<size_t N, typename T>
bool transf_list<N, T>::is_found(const transf<N, T> &tr) const {

	return std::find(m_trlist.begin(), m_trlist.end(), tr) !=
		m_trlist.end();
}


template<size_t N, typename T>
void transf_list<N, T>::make_list(const symmetry<N, T> &sym,
	const index<N> &idx, const transf<N, T> &tr) {

	if(is_found(tr)) return;

	m_trlist.push_back(tr);

	for(typename symmetry<N, T>::iterator iset = sym.begin();
		iset != sym.end(); iset++) {

		const symmetry_element_set<N, T> &eset = sym.get_subset(iset);
		for(typename symmetry_element_set<N, T>::const_iterator ielem =
			eset.begin(); ielem != eset.end(); ielem++) {

			const symmetry_element_i<N, T> &elem =
				eset.get_elem(ielem);
			index<N> idx2(idx);
			transf<N, T> tr2(tr);
			elem.apply(idx2, tr2);
			if(idx2.equals(idx)) make_list(sym, idx2, tr2);
		}
	}
}


} // namespace libtensor

#endif // LIBTENSOR_TRANSF_LIST_H
