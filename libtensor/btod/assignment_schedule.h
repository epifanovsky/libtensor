#ifndef LIBTENSOR_ASSIGNMENT_SCHEDULE_H
#define LIBTENSOR_ASSIGNMENT_SCHEDULE_H

#include <set>
#include <vector>
#include "../core/abs_index.h"
#include "../core/orbit_list.h"

namespace libtensor {


/**	\brief Contains an ordered schedule for the blockwise assignment (copy)
	\tparam N Tensor order.
	\tparam T Tensor element type.

	This container stores an ordered list of block indexes. Each %index
	specifies a non-zero block to be evaluated. The order of indexes is
	the same in which indexes are put on the list using insert().
	It should usually be the preferred order of evaluating the blocks.

	\ingroup libtensor_btod
 **/
template<size_t N, typename T>
class assignment_schedule {
public:
	typedef typename std::vector<size_t>::const_iterator iterator;

private:
	dimensions<N> m_bidims; //!< Block %index %dimensions
	std::vector<size_t> m_sch; //!< Schedule
	std::set<size_t> m_set; //!< Set of indexes

public:
	/**	\brief Creates an empty schedule
	 **/
	assignment_schedule(const dimensions<N> &bidims) : m_bidims(bidims) { }

	/**	\brief Creates a schedule containing canonical indexes in the
			ascending order
	 **/
	assignment_schedule(const symmetry<N, T> &sym);

	/**	\brief Appends an %index to the end of the list
	 **/
	void insert(const index<N> &idx);

	/**	\brief Checks whether the schedule contains a particular %index
	 **/
	bool contains(const index<N> &idx) const;

	/**	\brief Checks whether the schedule contains a particular %index
			by its absolute value
	 **/
	bool contains(size_t idx) const;

	/**	\brief Returns the iterator pointing at the first element
	 **/
	iterator begin() const {
		return m_sch.begin();
	}

	/**	\brief Returns the iterator pointing at the position after the
			last element
	 **/
	iterator end() const {
		return m_sch.end();
	}

	/**	\brief Returns the absolute value of the %index corresponding
			to an iterator
	 **/
	size_t get_abs_index(const iterator &i) const {
		return *i;
	}

private:
	void insert(size_t idx);

};


template<size_t N, typename T>
assignment_schedule<N, T>::assignment_schedule(const symmetry<N, T> &sym) :
	m_bidims(sym.get_bis().get_block_index_dims()) {

	orbit_list<N, T> ol(sym);
	for(typename orbit_list<N, T>::iterator i = ol.begin(); i != ol.end();
		i++) {

		insert(ol.get_abs_index(i));
	}
}


template<size_t N, typename T>
void assignment_schedule<N, T>::insert(const index<N> &idx) {

	abs_index<N> aidx(idx, m_bidims);
	insert(aidx.get_abs_index());
}


template<size_t N, typename T>
inline void assignment_schedule<N, T>::insert(size_t idx) {

	m_sch.push_back(idx);
	m_set.insert(idx);
}


template<size_t N, typename T>
bool assignment_schedule<N, T>::contains(const index<N> &idx) const {

	abs_index<N> aidx(idx, m_bidims);
	return contains(aidx.get_abs_index());
}


template<size_t N, typename T>
inline bool assignment_schedule<N, T>::contains(size_t idx) const {

	return m_set.find(idx) != m_set.end();
}


} // namespace libtensor

#endif // LIBTENSOR_ASSIGNMENT_SCHEDULE_H
