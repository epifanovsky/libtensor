#ifndef LIBTENSOR_BTOD_SELECT_H
#define LIBTENSOR_BTOD_SELECT_H

#include "../defs.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../symmetry/so_copy.h"
#include "../tod/tod_select.h"


namespace libtensor {

/**	\brief Selects a number of elements from a block %tensor
	\tparam N Tensor order.
	\tparam ComparePolicy Policy to select elements.

	The operation selects a number of elements from a block %tensor and adds
	them as (block %index, %index, value) data to a given list. If a symmetry
	is given, this symmetry is imposed on the block tensor to select elements
	from unique, non-zero blocks within this symmetry (the block tensor is not
	modified by this). The elements	are selected by the ordering imposed on the
	elements by the compare policy. Zero elements are never selected. The
	resulting list of elements is ordered according to the compare policy.

	TODO: The operation does not take proper care of blocks in an orbit whose
	transformation coefficient is -1.0, if the compare policy does not compare
	the absolute values. For the moment we do not need such select routine.
	Still this should be fixed in future.

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
	block_tensor_i<N, double> &m_bt; //!< Block tensor to select data from
	symmetry<N, double> m_sym; //!< Symmetry imposed on block tensor
	compare_t m_cmp; //!< Compare policy object to select entries

public:
	//! \name Constructor/destructor
	//@{

	/** \brief Constuctor without specific symmetry
	 	\param bt Block %tensor
		\param cmp Compare policy object (default: compare4absmin)
	 **/
	btod_select(block_tensor_i<N, double> &bt, compare_t cmp=compare_t());

	/** \brief Constuctor using symmetry
	 	\param bt Block %tensor
	 	\param sym Symmetry
		\param cmp Compare policy object (default: compare4absmin)
	 **/
	btod_select(block_tensor_i<N, double> &bt,
			const symmetry<N, double> &sym, compare_t cmp=compare_t());

	//@}

	/**	\brief Performs the operation
		\param bt Block %tensor.
		\param li List of elements.
		\param n Maximum list size.
	 **/
	void perform(list_t &li, size_t n);

private:
	btod_select(const btod_select<N, ComparePolicy> &);
	const btod_select<N> &operator=(const btod_select<N, ComparePolicy> &);

};

template<size_t N, typename ComparePolicy>
const char *btod_select<N,ComparePolicy>::k_clazz =
		"btod_select<N,ComparePolicy>";


template<size_t N, typename ComparePolicy>
btod_select<N, ComparePolicy>::btod_select(
		block_tensor_i<N, double> &bt, compare_t cmp) :
	m_bt(bt), m_sym(m_bt.get_bis()), m_cmp(cmp) {

	block_tensor_ctrl<N, double> ctrl(m_bt);
	so_copy<N, double>(ctrl.req_const_symmetry()).perform(m_sym);

}

template<size_t N, typename ComparePolicy>
btod_select<N, ComparePolicy>::btod_select(block_tensor_i<N, double> &bt,
		const symmetry<N, double> &sym, compare_t cmp) :
	m_bt(bt), m_sym(m_bt.get_bis()), m_cmp(cmp) {

	static const char *method =
			"btod_select(block_tensor_i<N, double>,	const symmetry<N, double> &, compare_t)";

	if (! m_sym.get_bis().equals(sym.get_bis()))
		throw bad_parameter(g_ns, k_clazz, method,
				__FILE__, __LINE__, "Invalid symmetry.");

	so_copy<N, double>(sym).perform(m_sym);
}


template<size_t N, typename ComparePolicy>
void btod_select<N, ComparePolicy>::perform(list_t &li, size_t n) {

	if (n == 0) return;

	block_tensor_ctrl<N, double> ctrl(m_bt);
	const block_index_space<N> &bis = m_bt.get_bis();
	const symmetry<N, double> &sym = ctrl.req_const_symmetry();

	orbit_list<N, double> ol(m_sym);

	for (typename orbit_list<N, double>::iterator iorb = ol.begin();
			iorb != ol.end(); iorb++) {

		index<N> blidx(ol.get_index(iorb));
		orbit<N, double> orb(sym, blidx);

		if (! orb.is_allowed()) continue;

		abs_index<N> cidx(orb.get_abs_canonical_index(),
				bis.get_block_index_dims());

		if (ctrl.req_is_zero_block(cidx.get_index())) continue;

		const transf<N, double> &tr = orb.get_transf(blidx);

		tensor_i<N, double> &t = ctrl.req_block(cidx.get_index());

		typedef tod_select<N, ComparePolicy> tselect_t;

		typename tselect_t::list_t tlist;
		tselect_t(t, m_cmp).perform(tlist, n);

		typename list_t::iterator ibt = li.begin();
		while (! tlist.empty()) {

			typename tselect_t::elem_t &el = tlist.front();

			el.value *= tr.get_coeff();
			el.idx.permute(tr.get_perm());
			while (ibt != li.end()) {
				if (m_cmp(el.value, ibt->value)) break;
				ibt++;
			}
			if (li.size() == n && ibt == li.end()) {
				tlist.clear();
			}
			else {
				ibt = li.insert(ibt, elem_t(blidx, el.idx, el.value));
				if (li.size() > n) li.pop_back();
				tlist.pop_front();
				ibt++;
			}
		}

		ctrl.ret_block(cidx.get_index());
	}
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SELECT_H
