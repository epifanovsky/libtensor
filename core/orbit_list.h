#ifndef LIBTENSOR_ORBIT_LIST_H
#define LIBTENSOR_ORBIT_LIST_H

#include <map>
#include <vector>
#include "defs.h"
#include "exception.h"
#include "timings.h"
#include "dimensions.h"
#include "index.h"
#include "symmetry.h"

namespace libtensor {


template<size_t N, typename T>
class orbit_list : public timings<orbit_list<N, T> > {
	friend class timings<orbit_list<N, T> >;
	static const char* k_clazz;
public:
	typedef typename std::map< size_t, index<N> >::const_iterator iterator;

private:
	dimensions<N> m_dims;
	std::map< size_t, index<N> > m_orb;

public:
	orbit_list(const symmetry<N, T> &sym);
	size_t get_size() const;
	bool contains(const index<N> &idx) const;
	bool contains(size_t absidx) const;
	iterator begin() const;
	iterator end() const;
	size_t get_abs_index(iterator &i) const;
	const index<N> &get_index(iterator &i) const;

private:
	bool mark_orbit(const symmetry<N, T> &sym, const index<N> &idx,
		std::vector<bool> &lst);
};

template<size_t N, typename T>
const char* orbit_list<N,T>::k_clazz="orbit_list<N,T>";

template<size_t N, typename T>
orbit_list<N, T>::orbit_list(const symmetry<N, T> &sym)
: m_dims(sym.get_bis().get_block_index_dims()) {
	orbit_list<N,T>::start_timer();
	
	std::vector<bool> chk(m_dims.get_size(), false);
	index<N> idx;
	do {
		size_t absidx = m_dims.abs_index(idx);
		if(!chk[absidx]) {
			if(mark_orbit(sym, idx, chk)) {
				m_orb.insert(std::pair< size_t, index<N> >(
					absidx, idx));
			}
		}
	} while(m_dims.inc_index(idx));

	orbit_list<N,T>::stop_timer();
}


template<size_t N, typename T>
inline size_t orbit_list<N, T>::get_size() const {

	return m_orb.size();
}


template<size_t N, typename T>
inline bool orbit_list<N, T>::contains(const index<N> &idx) const {

	return contains(m_dims.abs_index(idx));
}


template<size_t N, typename T>
inline bool orbit_list<N, T>::contains(size_t absidx) const {

	return m_orb.find(absidx) != m_orb.end();
}


template<size_t N, typename T>
inline typename orbit_list<N, T>::iterator orbit_list<N, T>::begin() const {

	return m_orb.begin();
}


template<size_t N, typename T>
inline typename orbit_list<N, T>::iterator orbit_list<N, T>::end() const {

	return m_orb.end();
}


template<size_t N, typename T>
inline size_t orbit_list<N, T>::get_abs_index(iterator &i) const {

	return i->first;
}


template<size_t N, typename T>
inline const index<N> &orbit_list<N, T>::get_index(iterator &i) const {

	return i->second;
}


template<size_t N, typename T>
bool orbit_list<N, T>::mark_orbit(const symmetry<N, T> &sym,
	const index<N> &idx, std::vector<bool> &lst) {

	size_t absidx = m_dims.abs_index(idx);
	bool allowed = true;
	if(!lst[absidx]) {
		lst[absidx] = true;
		size_t nelem = sym.get_num_elements();
		for(size_t ielem = 0; ielem < nelem; ielem++) {
			const symmetry_element_i<N, T> &elem =
				sym.get_element(ielem);
			allowed = allowed && elem.is_allowed(idx);
			index<N> idx2(idx);
			elem.apply(idx2);
			allowed = allowed && mark_orbit(sym, idx2, lst);
		}
	}
	return allowed;
}



} // namespace libtensor

#endif // LIBTENSOR_ORBIT_LIST_H
