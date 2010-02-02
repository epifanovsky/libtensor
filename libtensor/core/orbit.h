#ifndef LIBTENSOR_ORBIT_H
#define LIBTENSOR_ORBIT_H

#include <map>
#include <utility>
#include <vector>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "transf.h"
#include "symmetry.h"

namespace libtensor {

template<size_t N, typename T> class symmetry;


/**	\brief Symmetry-equivalent blocks of a block %tensor

	The action of the %index %symmetry group on the set of all block
	indexes in a block %tensor generates an %index set partition, each
	subset being an orbit. The smallest %index in an orbit is its canonical
	%index. The block %tensor shall only keep the block that corresponds
	to the canonical %index. All the blocks that are connected with the
	canonical block via %symmetry elements can be obtained by applying
	a transformation to the canonical block.

	<b>Orbit evaluation</b>
	<b>Orbit iterator</b>

	\ingroup libtensor_core
 **/
template<size_t N, typename T>
class orbit : public timings<orbit<N, T> > {
	friend class timings<orbit<N, T> >;
	static const char* k_clazz;
public:
	typedef typename std::map< size_t, transf<N, T> >::const_iterator
		iterator;

private:
	typedef std::pair< size_t, transf<N, T> > pair_t;
	typedef std::map< size_t, transf<N, T> > orbit_map_t;

private:
	dimensions<N> m_dims;
	orbit_map_t m_orb; //!< Orbit indexes
	size_t m_canidx; //!< Absolute %index of the canonical element

public:
	orbit(const symmetry<N, T> &sym, const index<N> &idx);
	size_t get_abs_canonical_index() const;
	size_t get_size() const;
	const transf<N, T> &get_transf(const index<N> &idx) const;
	const transf<N, T> &get_transf(size_t absidx) const;

	//!	\name STL-like iterator
	//@{
	iterator begin() const;
	iterator end() const;
	size_t get_abs_index(iterator &i) const;
	const transf<N, T> &get_transf(iterator &i) const;
	//@}

private:
	void mark_orbit(const symmetry<N, T> &sym, const index<N> &idx,
		std::vector<char> &lst, const transf<N, T> &tr);
};

template<size_t N, typename T>
const char* orbit<N, T>::k_clazz="orbit<N, T>";

template<size_t N, typename T>
orbit<N, T>::orbit(const symmetry<N, T> &sym, const index<N> &idx)
: m_dims(sym.get_bis().get_block_index_dims()) {
	orbit<N, T>::start_timer();

	m_canidx = m_dims.abs_index(idx);
	std::vector<char> chk(m_dims.get_size(), 0);
	transf<N, T> tr;
	mark_orbit(sym, idx, chk, tr);

	transf<N, T> tr_can(get_transf(m_canidx));
	tr_can.invert();
	typename orbit_map_t::iterator i = m_orb.begin();
	while(i != m_orb.end()) {
		i->second.transform(tr_can);
		i++;
	}

	orbit<N, T>::stop_timer();
}


template<size_t N, typename T>
inline size_t orbit<N, T>::get_abs_canonical_index() const {

	return m_canidx;
}


template<size_t N, typename T>
inline size_t orbit<N, T>::get_size() const {

	return m_orb.size();
}


template<size_t N, typename T>
inline const transf<N, T> &orbit<N, T>::get_transf(const index<N> &idx) const {

	return get_transf(m_dims.abs_index(idx));
}


template<size_t N, typename T>
inline const transf<N, T> &orbit<N, T>::get_transf(size_t absidx) const {

	typename orbit_map_t::const_iterator i = m_orb.find(absidx);
	return i->second;
}


template<size_t N, typename T>
inline typename orbit<N, T>::iterator orbit<N, T>::begin() const {

	return m_orb.begin();
}


template<size_t N, typename T>
inline typename orbit<N, T>::iterator orbit<N, T>::end() const {

	return m_orb.end();
}


template<size_t N, typename T>
inline size_t orbit<N, T>::get_abs_index(iterator &i) const {

	return i->first;
}


template<size_t N, typename T>
inline const transf<N, T> &orbit<N, T>::get_transf(iterator &i) const {

	return i->second;
}


template<size_t N, typename T>
void orbit<N, T>::mark_orbit(const symmetry<N, T> &sym, const index<N> &idx,
	std::vector<char> &lst, const transf<N, T> &tr) {

	size_t absidx = m_dims.abs_index(idx);
	if(absidx < m_canidx) m_canidx = absidx;
	if(lst[absidx] == 0) {
		lst[absidx] = 1;
		m_orb.insert(pair_t(absidx, tr));
		typename symmetry<N, T>::iterator ielem = sym.begin();
		for(; ielem != sym.end(); ielem++) {
			const symmetry_element_i<N, T> &elem =
				sym.get_element(ielem);
			index<N> idx2(idx);
			transf<N, T> tr2(tr);
			elem.apply(idx2, tr2);
			mark_orbit(sym, idx2, lst, tr2);
		}
	}
}


} // namespace libtensor

#endif // LIBTENSOR_ORBIT_H
