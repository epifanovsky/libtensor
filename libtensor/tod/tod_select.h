#ifndef LIBTENSOR_TOD_SELECT_H
#define LIBTENSOR_TOD_SELECT_H

#include "../defs.h"
#include "../exception.h"
#include "../core/abs_index.h"
#include "../core/index.h"
#include "../core/tensor_ctrl.h"
#include <cmath>

namespace libtensor {

//! \name Common compare policies
//@{

struct compare4max {
	bool operator()(const double &a, const double &b) {
		return a > b;
	}
};

struct compare4absmax {
	bool operator()(const double &a, const double &b) {
		return fabs(a) > fabs(b);
	}
};

struct compare4min {
	bool operator()(const double &a, const double &b) {
		return a < b;
	}
};

struct compare4absmin {
	bool operator()(const double &a, const double &b) {
		return fabs(a) < fabs(b);
	}
};
//@}


/**	\brief Selects a number of elements from a %tensor
	\tparam N Tensor order.
	\tparam ComparePolicy Policy to select elements.

	The operation selects a number of elements from the %tensor and adds them
	as (index, value) to a given list. The elements are selected by the
	ordering imposed on the elements by the compare policy. Zero elements are
	never selected. The resulting list of elements is ordered according to the
	compare policy.

	If a permutation and / or a coefficient are given in the construct, the
	tensor elements are permuted and scaled before the list is constructed
	(this does not affect the input tensor).

	<b>Compare policy</b>

	The compare policy type determines the ordering of %tensor elements by which
	they are selected. Any type used as compare policy needs to implement a
	function
	<code>
		bool operator(const double&, const double&)
	</code>
	which compares two %tensor elements. If the function returns true, the first
	value is taken as the more optimal with respect to the compare policy.

	\ingroup libtensor_tod
 **/
template<size_t N, typename ComparePolicy=compare4absmin>
class tod_select {
public:
	typedef ComparePolicy compare_t;

	struct elem {
		index<N> idx;
		double value;

		elem(const index<N> &idx_, const double &v=0.0) : idx(idx_), value(v) {}
	};
	typedef elem elem_t; //!< Tensor index-value pair type
	typedef std::list<elem_t> list_t; //!< List type for index-value pairs

private:
	dense_tensor_i<N, double> &m_t; //!< Tensor
	permutation<N> m_perm; //!< Permutation of tensor
	double m_c; //!< Scaling coefficient
	compare_t m_cmp; //!< Compare policy object to select entries

public:
	//! \name Constructor/destructor
	//@{

	/** \brief Constuctor
		\param t Tensor.
		\param cmp Compare policy object
	 **/
	tod_select(dense_tensor_i<N, double> &t, compare_t cmp = compare_t()) :
		m_t(t), m_c(1.0), m_cmp(cmp) { }

	/** \brief Constuctor
		\param t Tensor.
		\param c Coefficient.
		\param cmp Compare policy object.
	 **/
	tod_select(dense_tensor_i<N, double> &t,
			double c, compare_t cmp = compare_t()) :
		m_t(t), m_c(c), m_cmp(cmp) { }

	/** \brief Constuctor
		\param t Tensor
		\param p Permutation
		\param c Coefficient
		\param cmp Compare policy object.
	 **/
	tod_select(dense_tensor_i<N, double> &t, const permutation<N> &p,
			double c, compare_t cmp = compare_t()) :
		m_t(t), m_perm(p), m_c(c), m_cmp(cmp) { }

	//@}

	//!	\name Operation
	//@{

	/**	\brief Selects the index-value pairs from the tensor
		\param li List of index-value pairs.
		\param n Maximum size of the list.
	 **/
	void perform(list_t &li, size_t n);

	//@}
};


template<size_t N, typename ComparePolicy>
void tod_select<N,ComparePolicy>::perform(list_t &li, size_t n) {

	if (n == 0) return;

	tensor_ctrl<N, double> ctrl(m_t);
	const dimensions<N> &d = m_t.get_dims();
	const double *p = ctrl.req_const_dataptr();

	bool do_perm = !m_perm.is_identity();

	size_t i = 0;
	while (i < d.get_size() && p[i] == 0.0) i++;

	if (i == d.get_size()) {
		ctrl.ret_const_dataptr(p);
		return;	
	}

	if (li.empty()) {
	    abs_index<N> aidx(i, d);
		index<N> idx(aidx.get_index());
		if (do_perm) idx.permute(m_perm);
		li.insert(li.end(), elem_t(idx, m_c * p[i]));
		i++;
	}

	for (; i < d.get_size(); i++) {
		//ignore zero elements
		if (p[i] == 0.0) continue;

		double val = p[i] * m_c;

		if (! m_cmp(val, li.back().value)) {
			if (li.size() < n) {
		        abs_index<N> aidx(i, d);
				index<N> idx(aidx.get_index());
				if (do_perm) idx.permute(m_perm);
				li.push_back(elem_t(idx, val));
			}
		}
		else {
			if (li.size() == n) li.pop_back();

			typename list_t::iterator it = li.begin();
			while (it != li.end() && ! m_cmp(val, it->value)) it++;
            abs_index<N> aidx(i, d);
			index<N> idx(aidx.get_index());
			if (do_perm) idx.permute(m_perm);
			li.insert(it, elem_t(idx, val));
		}
	}

	ctrl.ret_const_dataptr(p);
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_SELECT_H
