#ifndef LIBTENSOR_ORBIT_H
#define LIBTENSOR_ORBIT_H

#include <map>
#include <utility>
#include <vector>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "abs_index.h"
#include "transf.h"
#include "symmetry.h"

namespace libtensor {


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
public:
	static const char *k_clazz; //!< Class name

public:
	typedef typename std::map< size_t, transf<N, T> >::const_iterator
		iterator; //!< Orbit iterator

private:
	typedef std::pair< size_t, transf<N, T> > pair_t;
	typedef std::map< size_t, transf<N, T> > orbit_map_t;

private:
	dimensions<N> m_bidims; //!< Block %index %dimensions
	orbit_map_t m_orb; //!< Map of %orbit indexes to transformations
	bool m_allowed; //!< Whether the orbit is allowed by %symmetry

public:
	/**	\brief Constructs the %orbit using a %symmetry group and
			any %index in the %orbit
	 **/
	orbit(const symmetry<N, T> &sym, const index<N> &idx);

	/**	\brief Returns whether the %orbit is allowed by %symmetry
	 **/
	bool is_allowed() const {

		return m_allowed;
	}

	/**	\brief Returns the canonical %index of this %orbit
	 **/
	size_t get_abs_canonical_index() const {

		return m_orb.begin()->first;
	}

	/**	\brief Returns the number of indexes in the orbit
	 **/
	size_t get_size() const {

		return m_orb.size();
	}

	/** \brief Obtain transformation of canonical block to yield block at idx.
		@param idx Block index
		@return Transformation to obtain the block at idx from the canonical block
	 **/
	const transf<N, T> &get_transf(const index<N> &idx) const;

	/** \brief Obtain transformation of canonical block to yield block at absidx.
		@param absidx Absolute block index
		@return Transformation to yield block at absidx
	 **/
	const transf<N, T> &get_transf(size_t absidx) const;

	/** \brief Checks if orbit contains block at idx.
		@param idx Block index
		@return True if orbit contains the block
	 **/
	bool contains(const index<N> &idx) const;

	/** \brief Checks if orbit contains block at absidx.
		@param absidx Absolute block index
		@return True if orbit contains the block
	 **/
	bool contains(size_t absidx) const;

	//!	\name STL-like %orbit iterator
	//@{

	iterator begin() const {

		return m_orb.begin();
	}

	iterator end() const {

		return m_orb.end();
	}

	size_t get_abs_index(iterator &i) const;

	const transf<N, T> &get_transf(iterator &i) const;

	//@}

private:
	void build_orbit(const symmetry<N, T> &sym, const index<N> &idx);

};


template<size_t N, typename T>
const char *orbit<N, T>::k_clazz = "orbit<N, T>";


template<size_t N, typename T>
orbit<N, T>::orbit(const symmetry<N, T> &sym, const index<N> &idx) :

	m_bidims(sym.get_bis().get_block_index_dims()) {

	orbit<N, T>::start_timer();

	m_allowed = true;
	build_orbit(sym, idx);

	orbit<N, T>::stop_timer();
}


template<size_t N, typename T>
inline const transf<N, T> &orbit<N, T>::get_transf(const index<N> &idx) const {

	return get_transf(abs_index<N>(idx, m_bidims).get_abs_index());
}


template<size_t N, typename T>
inline const transf<N, T> &orbit<N, T>::get_transf(size_t absidx) const {

	iterator i = m_orb.find(absidx);
	return get_transf(i);
}

template<size_t N, typename T>
inline bool orbit<N, T>::contains(const index<N> &idx) const {

	return contains(abs_index<N>(idx, m_bidims).get_abs_index());
}


template<size_t N, typename T>
inline bool orbit<N, T>::contains(size_t absidx) const {

	return m_orb.find(absidx) != m_orb.end();
}

template<size_t N, typename T>
inline size_t orbit<N, T>::get_abs_index(iterator &i) const {

	static const char *method = "get_abs_index(iterator&)";

#ifdef LIBTENSOR_DEBUG
	if(i == m_orb.end()) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"i");
	}
#endif // LIBTENSOR_DEBUG

	return i->first;
}


template<size_t N, typename T>
inline const transf<N, T> &orbit<N, T>::get_transf(iterator &i) const {

	static const char *method = "get_transf(iterator&)";

#ifdef LIBTENSOR_DEBUG
	if(i == m_orb.end()) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"i");
	}
#endif // LIBTENSOR_DEBUG

	return i->second;
}


template<size_t N, typename T>
void orbit<N, T>::build_orbit(const symmetry<N, T> &sym, const index<N> &idx) {

	std::vector< index<N> > qi;
	std::vector< transf<N, T> > qt;
	std::vector< index<N> > ti;
	std::vector< transf<N, T> > tt;

	qi.reserve(32);
	qt.reserve(32);
	ti.reserve(32);
	tt.reserve(32);

	abs_index<N> aidx0(idx, m_bidims);
	m_orb.insert(pair_t(aidx0.get_abs_index(), transf<N, T>()));

	qi.push_back(idx);
	qt.push_back(transf<N, T>());

	while(!qi.empty()) {

		index<N> idx1(qi.back());
		transf<N, T> tr1(qt.back());
		qi.pop_back();
		qt.pop_back();

		for(typename symmetry<N, T>::iterator iset = sym.begin();
			iset != sym.end(); iset++) {

			const symmetry_element_set<N, T> &eset =
				sym.get_subset(iset);
			for(typename symmetry_element_set<N, T>::const_iterator
				ielem = eset.begin(); ielem != eset.end();
				ielem++) {

				const symmetry_element_i<N, T> &elem =
					eset.get_elem(ielem);
				m_allowed = m_allowed && elem.is_allowed(idx1);
				ti.push_back(idx1);
				tt.push_back(tr1);
				elem.apply(ti.back(), tt.back());
			}
		}
		for(size_t i = 0; i < ti.size(); i++) {
			abs_index<N> aidx(ti[i], m_bidims);
			if(m_orb.insert(pair_t(aidx.get_abs_index(),
				tt[i])).second) {
				qi.push_back(ti[i]);
				qt.push_back(tt[i]);
			}
		}
		ti.clear();
		tt.clear();
	}

	transf<N, T> tr0(m_orb.begin()->second);
	tr0.invert();
	for(typename orbit_map_t::iterator i = m_orb.begin();
		i != m_orb.end(); i++) {
		i->second.transform(tr0);
	}
}


} // namespace libtensor

#endif // LIBTENSOR_ORBIT_H
