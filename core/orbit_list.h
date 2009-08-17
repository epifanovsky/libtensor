#ifndef LIBTENSOR_ORBIT_LIST_H
#define LIBTENSOR_ORBIT_LIST_H

#include <vector>
#include "defs.h"
#include "exception.h"
#include "index.h"
#include "symmetry.h"

namespace libtensor {


template<size_t N, typename T>
class orbit_list {
public:
	typedef typename std::vector< index<N> >::const_iterator iterator;

private:
	std::vector< index<N> > m_orb;

public:
	orbit_list(const symmetry<N, T> &sym);
	size_t get_size() const;
	iterator begin() const;
	iterator end() const;

private:
	bool mark_orbit(const symmetry<N, T> &sym, const index<N> &idx,
		std::vector<bool> &lst);
};


template<size_t N, typename T>
orbit_list<N, T>::orbit_list(const symmetry<N, T> &sym) {

	const dimensions<N> &dims = sym.get_dims();
	std::vector<bool> chk(dims.get_size(), false);
	index<N> idx;
	do {
		size_t absidx = dims.abs_index(idx);
		if(!chk[absidx]) {
			if(mark_orbit(sym, idx, chk)) m_orb.push_back(idx);
		}
	} while(dims.inc_index(idx));
}


template<size_t N, typename T>
inline size_t orbit_list<N, T>::get_size() const {

	return m_orb.size();
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
bool orbit_list<N, T>::mark_orbit(const symmetry<N, T> &sym,
	const index<N> &idx, std::vector<bool> &lst) {

	size_t absidx = sym.get_dims().abs_index(idx);
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
