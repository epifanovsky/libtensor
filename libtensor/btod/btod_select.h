#ifndef LIBTENSOR_BTOD_SELECT_H
#define LIBTENSOR_BTOD_SELECT_H

#include "../defs.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit_list.h"
#include "../tod/tod_select.h"


namespace libtensor {

/**	\brief Selects a number of elements from a block %tensor
	\tparam N Tensor order.
	\tparam ComparePolicy Policy to select elements.

	The operation selects a number of elements from a block %tensor and adds
	them as (block %index, %index, value) data to a given list. The elements
	are selected by the ordering imposed on the elements by the compare policy.
	Zero elements are never selected. The resulting list of elements is ordered
	according to the compare policy.

	<b>Compare policy</b>

	The compare policy type determines the ordering of block %tensor elements by
	which they are selected. Any type used as compare policy needs to implement
	a function
	<code>
		bool operator( const double&, const double& )
	</code>
	which compares two block %tensor elements. If the function returns true,
	the first value is taken to be more optimal with respect to the compare
	policy.

	\ingroup libtensor_btod
 **/
template<size_t N, typename ComparePolicy=compare4absmin>
class btod_select {
public:
	static const char *k_clazz; //!< Class name

	typedef ComparePolicy compare_t;

	struct elem {
		index<N> bidx;
		index<N> idx;
		double	value;

		elem( const index<N> &bidx_, const index<N> &idx_,
				const double& v ) : bidx(bidx_), idx(idx_), value(v) {}
	};
	typedef elem elem_t; //!< Tensor index-value pair type
	typedef std::list<elem_t> list_t; //!< List type for index-value pairs

private:
	compare_t m_cmp; //!< Compare policy object to select entries

public:
	//! \name Constructor/destructor
	//@{

	/** \brief Default constuctor
		\param cmp Compare policy object (default: compare4absmin)
	 **/
	btod_select( compare_t cmp=compare4absmin() ) : m_cmp(cmp) {}

	//@}

	/**	\brief Performs the operation
		\param bt Block %tensor.
		\param li List of elements.
		\param n Maximum list size.
	 **/
	void perform(block_tensor_i<N, double> &bt, list_t &li, size_t n);

private:
	btod_select(const btod_select<N, ComparePolicy> &);
	const btod_select<N> &operator=(const btod_select<N, ComparePolicy> &);

};


template<size_t N, typename ComparePolicy>
const char *btod_select<N,ComparePolicy>::k_clazz =
		"btod_select<N,ComparePolicy>";

template<size_t N, typename ComparePolicy>
void btod_select<N,ComparePolicy>::perform(block_tensor_i<N, double> &bt,
	list_t &li, size_t n) {

	if (n == 0) return;

	block_tensor_ctrl<N, double> ctrl(bt);
	orbit_list<N, double> orblist(ctrl.req_symmetry());

	typedef tod_select<N, ComparePolicy> tselect_t;
	tselect_t tselect(m_cmp);
	typename tselect_t::list_t tlist;

	for (typename orbit_list<N, double>::iterator iorb=orblist.begin();
			iorb != orblist.end(); iorb++) {

		index<N> blidx(orblist.get_index(iorb));
		if (ctrl.req_is_zero_block(blidx)) continue;

		tensor_i<N, double> &t = ctrl.req_block(blidx);
		tselect.perform(t,tlist,n);

		typename list_t::iterator ibt = li.begin();
		while (! tlist.empty()) {
			typename tselect_t::elem_t &el = tlist.front();
			while ( ibt != li.end() ) {
				if ( m_cmp(el.value,ibt->value) ) break;
				ibt++;
			}
			if ( li.size() == n && ibt == li.end() ) {
				tlist.clear();
			}
			else {
				ibt=li.insert(ibt, elem_t(blidx,el.idx,el.value));
				if ( li.size() > n ) li.pop_back();
				tlist.pop_front();
				ibt++;
			}
		}

		ctrl.ret_block(blidx);
	}
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SELECT_H
