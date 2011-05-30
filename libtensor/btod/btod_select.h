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

	The operation uses a block %tensor, a %symmetry and a compare policy to
	create an ordered list of block %tensor elements as (block %index,
	%index, value) data. The optional %symmetry is employed to determine the
	blocks from which elements can be selected. If it is not given, the
	internal symmetry of the block %tensor is used instead.

	Elements are selected exclusively from blocks which are unique, allowed,
	and non-zero within the given symmetry. If the symmetry by which the
	blocks are determined differs from the %symmetry of the block %tensor,
	the unique blocks within both symmetries might differ. Is this the case,
	a block present in the block %tensor might be transformed to yield the
	unique block within the symmetry before elements are selected.

	<b>Compare policy</b>

	The compare policy type determines the ordering of block %tensor elements
	by which they are selected. Any type used as compare policy needs to
	implement a function
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
	typedef tod_select<N, compare_t> tod_select_t;
	typedef typename tod_select_t::list_t tod_list_t;
	typedef typename tod_select_t::elem_t tod_elem_t;

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
	void merge_lists(list_t &to, const index<N> &bidx,
			const tod_list_t &from, size_t n);

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
			"btod_select(block_tensor_i<N, double>,	"
			"const symmetry<N, double> &, compare_t)";

	if (! m_sym.get_bis().equals(sym.get_bis()))
		throw bad_parameter(g_ns, k_clazz, method,
				__FILE__, __LINE__, "Invalid symmetry.");

	so_copy<N, double>(sym).perform(m_sym);
}


template<size_t N, typename ComparePolicy>
void btod_select<N, ComparePolicy>::perform(list_t &li, size_t n) {

	static const char *method = "perform(list_t &, size_t)";

	if (n == 0) return;
	li.clear();

	const block_index_space<N> &bis = m_bt.get_bis();
	dimensions<N> bidims(bis.get_block_index_dims());

	block_tensor_ctrl<N, double> ctrl(m_bt);
	const symmetry<N, double> &sym = ctrl.req_const_symmetry();

	// Loop over all orbits of imposed symmetry
	orbit_list<N, double> ol(m_sym);
	for (typename orbit_list<N, double>::iterator iol = ol.begin();
			iol != ol.end(); iol++) {

		abs_index<N> aia(ol.get_abs_index(iol), bidims);

		orbit<N, double> oa(sym, aia.get_index());
		if (! oa.is_allowed()) continue;

		abs_index<N> aib(oa.get_abs_canonical_index(), bidims);
		if (ctrl.req_is_zero_block(aib.get_index())) continue;

		// Obtain block
		tensor_i<N, double> &t = ctrl.req_block(aib.get_index());

		const transf<N, double> &tra = oa.get_transf(aia.get_index());

		// Create element list for canonical block (within the symmetry)
		tod_list_t tlc;
		tod_select_t(t, tra.get_perm(), tra.get_coeff(), m_cmp).perform(tlc, n);
		merge_lists(li, aia.get_index(), tlc, n);

//		// Loop over whole orbit of imposed symmetry
//		orbit<N, double> ob(m_sym, aia.get_index());
//		for (typename orbit<N, double>::iterator iob = ob.begin();
//				iob != ob.end(); iob++) {
//
//			abs_index<N> aic(ob.get_abs_index(iob), bidims);
//			const transf<N, double> &trb = ob.get_transf(iob);
//
//			// Same index, identity transformation: skip block
//			if (aia.get_abs_index() == aic.get_abs_index() &&
//					trb.is_identity() && trb.get_coeff() == 1.0)
//				continue;
//
//			// Identity transformation: add elements under different block
//			if (trb.is_identity() && trb.get_coeff() == 1.0) {
//				merge_lists(li, aic.get_index(), tlc, n);
//			}
//			// Permutation: add permuted elements
//			else {
//				permutation<N> pa(tra.get_perm());
//				pa.permute(trb.get_perm());
//
//				tod_list_t tlx;
//				tod_select_t(t, pa, tra.get_coeff() * trb.get_coeff(),
//						m_cmp).perform(tlx, n);
//
//				merge_lists(li, aic.get_index(), tlx, n);
//			}
//		}
		ctrl.ret_block(aib.get_index());
	}
}

template<size_t N, typename ComparePolicy>
void btod_select<N, ComparePolicy>::merge_lists(
		list_t &to, const index<N> &bidx, const tod_list_t &from, size_t n) {

	typename list_t::iterator ibt = to.begin();
	for (typename tod_list_t::const_iterator it = from.begin();
			it != from.end(); it++) {

		while (ibt != to.end()) {
			if (m_cmp(it->value, ibt->value)) break;
			ibt++;
		}

		if (to.size() == n && ibt == to.end()) {
			return;
		}

		ibt = to.insert(ibt, elem_t(bidx, it->idx, it->value));
		if (to.size() > n) to.pop_back();
		ibt++;
	}
}

} // namespace libtensor

#endif // LIBTENSOR_BTOD_SELECT_H
