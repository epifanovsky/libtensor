#ifndef LIBTENSOR_ASSIGNMENT_SCHEDULE_H
#define LIBTENSOR_ASSIGNMENT_SCHEDULE_H

#include <list>
#include "../core/abs_index.h"

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
	typedef typename std::list<size_t>::const_iterator iterator;

private:
	dimensions<N> m_bidims; //!< Block %index %dimensions
	std::list<size_t> m_sch; //!< Schedule

public:
	/**	\brief Creates an empty schedule
	 **/
	assignment_schedule(const dimensions<N> &bidims) : m_bidims(bidims) { }

	/**	\brief Appends an %index to the end of the list
	 **/
	void insert(const index<N> &idx);

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

};


template<size_t N, typename T>
void assignment_schedule<N, T>::insert(const index<N> &idx) {

	abs_index<N> aidx(idx, m_bidims);
	m_sch.push_back(aidx.get_abs_index());
}


} // namespace libtensor

#endif // LIBTENSOR_ASSIGNMENT_SCHEDULE_H
