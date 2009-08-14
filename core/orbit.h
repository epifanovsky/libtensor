#ifndef LIBTENSOR_ORBIT_H
#define LIBTENSOR_ORBIT_H

#include <utility>
#include <vector>
#include "defs.h"
#include "exception.h"
#include "transf.h"
#include "symmetry.h"

namespace libtensor {

template<size_t N, typename T> class symmetry;


template<size_t N, typename T>
class orbit {
private:
	typedef std::pair< size_t, transf<N, T> > record_t;

private:
	const symmetry<N, T> &m_symmetry; //!< Reference to parent symmetry
	size_t m_absidx; //!< Absolute %index of the canonical element
	mutable std::vector<record_t> m_orb;
	mutable bool m_dirty;

public:
	orbit(const symmetry<N, T> &sym, size_t absidx);
	size_t get_abs_canonical_index() const;
	size_t get_size() const;
	size_t get_abs_index(size_t n) const throw(out_of_bounds);
	const transf<N, T> &get_transf(size_t n) const throw(out_of_bounds);

private:
	void build() const;
	void mark_orbit(const dimensions<N> &dims, const index<N> &idx,
		std::vector<bool> &lst, transf<N, T> &tr) const;
};


template<size_t N, typename T>
inline orbit<N, T>::orbit(const symmetry<N, T> &sym, size_t absidx)
: m_symmetry(sym), m_absidx(absidx), m_dirty(true) {

}


template<size_t N, typename T>
inline size_t orbit<N, T>::get_abs_canonical_index() const {

	return m_absidx;
}


template<size_t N, typename T>
size_t orbit<N, T>::get_size() const {

	if(m_dirty) build();
	return m_orb.size();
}


template<size_t N, typename T>
size_t orbit<N, T>::get_abs_index(size_t n) const throw(out_of_bounds) {

	if(m_dirty) build();
	if(n >= m_orb.size()) {
		throw out_of_bounds("libtensor", "orbit<N, T>",
			"get_abs_index(size_t)", __FILE__, __LINE__,
			"Index number is out of bounds.");
	}
	return m_orb[n].first;
}


template<size_t N, typename T>
const transf<N, T> &orbit<N, T>::get_transf(size_t n) const
	throw(out_of_bounds) {

	if(m_dirty) build();
	if(n >= m_orb.size()) {
		throw out_of_bounds("libtensor", "orbit<N, T>",
			"get_transf(size_t)", __FILE__, __LINE__,
			"Index number is out of bounds.");
	}
	return m_orb[n].second;
}


template<size_t N, typename T>
void orbit<N, T>::build() const {

	m_orb.clear();

	const dimensions<N> &dims = m_symmetry.get_dims();
	std::vector<bool> chk(dims.get_size(), false);
	index<N> idx;
	dims.abs_index(m_absidx, idx);
	transf<N, T> tr;
	mark_orbit(dims, idx, chk, tr);

	typename std::vector<record_t>::iterator i = m_orb.begin();
	while(i != m_orb.end()) {
		i->second.invert();
		i++;
	}

	m_dirty = false;
}


template<size_t N, typename T>
void orbit<N, T>::mark_orbit(const dimensions<N> &dims, const index<N> &idx,
	std::vector<bool> &lst, transf<N, T> &tr) const {

	size_t absidx = dims.abs_index(idx);
	if(!lst[absidx]) {
		lst[absidx] = true;
		m_orb.push_back(record_t(absidx, tr));
		size_t nelem = m_symmetry.get_num_elements();
		for(size_t ielem = 0; ielem < nelem; ielem++) {
			const symmetry_element_i<N, T> &elem =
				m_symmetry.get_element(ielem);
			index<N> idx2(idx);
			transf<N, T> tr2(tr);
			elem.apply(idx2, tr2);
			mark_orbit(dims, idx2, lst, tr2);
		}
	}
}


} // namespace libtensor

#endif // LIBTENSOR_ORBIT_H
