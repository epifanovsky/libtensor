#ifndef LIBTENSOR_BLOCK_INDEX_MAP_H
#define LIBTENSOR_BLOCK_INDEX_MAP_H

#include <map>
#include "abs_index.h"
#include "block_index_map_i.h"
#include "out_of_bounds.h"

namespace libtensor {


/**	\brief Block %index map that stores mappings explicitly
	\tparam N Tensor order.

	Implementation of block_index_map_i<N> that stores block mappings
	explicitly. Initial setup is done by copying the mappings from
	another object that implements block_index_map_i<N>.

	\ingroup libtensor_core
 **/
template<size_t N>
class block_index_map : public block_index_map_i<N> {
public:
	static const char *k_clazz; //!< Class name

private:
	block_index_space<N> m_bis_from; //!< Source block %index space
	block_index_space<N> m_bis_to; //!< Destination block %index space
	dimensions<N> m_bidims_from; //!< Source block %index dims
	dimensions<N> m_bidims_to; //!< Destination block %index dims
	std::map<size_t, size_t> m_map; //!< Mapping

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Constructor
	 **/
	block_index_map(const block_index_map_i<N> &map);

	/**	\brief Virtual destructor
	 **/
	virtual ~block_index_map() { }

	//@}


	//!	\name Implementation of block_index_map_i<N, T>
	//@{

	virtual const block_index_space<N> &get_bis_from() const {
		return m_bis_from;
	}

	virtual const block_index_space<N> &get_bis_to() const {
		return m_bis_to;
	}

	virtual bool map_exists(const index<N> &from) const;

	virtual void get_map(const index<N> &from, index<N> &to) const;

	//@}

};


template<size_t N>
const char *block_index_map<N>::k_clazz = "block_index_map<N>";


template<size_t N>
block_index_map<N>::block_index_map(const block_index_map_i<N> &map) :

	m_bis_from(map.get_bis_from()), m_bis_to(map.get_bis_to()),
	m_bidims_from(m_bis_from.get_block_index_dims()),
	m_bidims_to(m_bis_to.get_block_index_dims()) {

	abs_index<N> aidx1(m_bidims_from);
	do {
		if(map.map_exists(aidx1.get_index())) {
			index<N> idx2;
			map.get_map(aidx1.get_index(), idx2);
			abs_index<N> aidx2(idx2, m_bidims_to);
			m_map[aidx1.get_abs_index()] = aidx2.get_abs_index();
		}
	} while(aidx1.inc());
}


template<size_t N>
bool block_index_map<N>::map_exists(const index<N> &from) const {

	abs_index<N> afrom(from, m_bidims_from);
	return m_map.find(afrom.get_abs_index()) != m_map.end();
}


template<size_t N>
void block_index_map<N>::get_map(const index<N> &from, index<N> &to) const {

	static const char *method = "get_map(const index<N>&, index<N>&)";

	abs_index<N> afrom(from, m_bidims_from);
	typename std::map<size_t, size_t>::const_iterator i =
		m_map.find(afrom.get_abs_index());
	if(i == m_map.end()) {
		throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
			"from");
	}
	abs_index<N> ato(i->second, m_bidims_to);
	to = ato.get_index();
}


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_INDEX_MAP_H
