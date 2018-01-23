#ifndef LIBUTIL_COND_MAP_H
#define LIBUTIL_COND_MAP_H

#include <map>
#include "loaded_cond.h"

namespace libutil {


/**	\brief Map of condition variables

	\ingroup libutil
 **/
template<typename KeyT, typename LoadT>
class cond_map {
private:
	typedef KeyT key_t; //!< Key type
	typedef LoadT load_t; //!< Load type
	typedef loaded_cond<load_t> cond_t; //!< Conditional variable type
	typedef std::multimap<key_t, cond_t*> map_t; //!< Map type
	typedef typename map_t::iterator iterator_t; //!< Iterator type

private:
	map_t m_map; //!< Map

public:
	/**	\brief Inserts a conditional
	 **/
	void insert(const key_t &key, cond_t *c) {

		m_map.insert(std::pair<key_t, cond_t*>(key, c));
	}

	/**	\brief Removes a conditional
	 **/
	void erase(const key_t &key, cond_t *c) {

		std::pair<iterator_t, iterator_t> irange =
			m_map.equal_range(key);
		for(iterator_t i = irange.first; i != irange.second; i++) {
			if(i->second == c) {
				m_map.erase(i);
				break;
			}
		}
	}

	/**	\brief Returns true if the map contains a key
	 **/
	bool contains(const key_t &key) {

		return m_map.count(key) != 0;
	}

	/**	\brief Signals all conditionals with a given key
	 **/
	void signal(const key_t &key) {

		std::pair<iterator_t, iterator_t> irange =
			m_map.equal_range(key);
		for(iterator_t i = irange.first; i != irange.second; i++) {
			i->second->signal();
		}
	}

	/**	\brief Sets load to all conditionals with a given key
	 **/
	void set_load(const key_t &key, const load_t &load) {

		std::pair<iterator_t, iterator_t> irange =
			m_map.equal_range(key);
		for(iterator_t i = irange.first; i != irange.second; i++) {
			i->second->load() = load;
		}
	}

};


} // namespace libutil

#endif // LIBUTIL_COND_MAP_H

