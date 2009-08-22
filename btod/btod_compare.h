#ifndef LIBTENSOR_BTOD_COMPARE_H
#define LIBTENSOR_BTOD_COMPARE_H

#include "defs.h"
#include "exception.h"
#include "core/block_tensor_ctrl.h"
#include "core/block_tensor_i.h"
#include "tod/tod_compare.h"

namespace libtensor {

/**	\brief Compares two block tensors

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_compare {
private:
	block_tensor_i<N, double> &m_bt1;
	block_tensor_i<N, double> &m_bt2;
	double m_thresh;
	index<N> m_idx_diff;
	double m_diff_elem_1, m_diff_elem_2;

public:
	/**	\brief Initializes the operation
		\param bt1 First %tensor.
		\param bt2 Second %tensor.
		\param thresh Threshold.

		The two block tensors must have compatible block index spaces,
		otherwise an exception will be thrown.
	**/
	btod_compare(block_tensor_i<N, double> &bt1,
		block_tensor_i<N, double> &bt2, double thresh) throw(exception);

	/**	\brief Performs the comparison
		\return \c true if all the elements equal within the threshold,
			\c false otherwise
		\throw Exception if the two tensors have different dimensions.
	**/
	bool compare();

	/**	\brief Returns the index of the first non-equal element
	**/
	const index<N> &get_diff_index() const;

	/**	\brief Returns the value of the first different element in
			the first block %tensor
	**/
	double get_diff_elem_1() const;

	/**	\brief Returns the value of the first different element in
			the second block %tensor
	**/
	double get_diff_elem_2() const;
};

template<size_t N>
inline btod_compare<N>::btod_compare(block_tensor_i<N, double> &bt1,
	block_tensor_i<N, double> &bt2, double thresh) throw(exception)
	: m_bt1(bt1), m_bt2(bt2), m_thresh(fabs(thresh)) {
	m_diff_elem_1 = 0.0; m_diff_elem_2 = 0.0;

	//const dimensions<N> &dim1(m_t1.get_dims()), &dim2(m_t2.get_dims());
	//for(size_t i=0; i<N; i++) if(dim1[i]!=dim2[i]) {
	//	throw_exc("tod_compare<N>", "tod_compare()",
	//		"The tensors have different dimensions");
	//}

}

template<size_t N>
bool btod_compare<N>::compare() {
	block_tensor_ctrl<N, double> ctrl1(m_bt1), ctrl2(m_bt2);
	
	orbit_list<N,double> orblist1(ctrl1.req_symmetry()),
		orblist2(ctrl2.req_symmetry());
	
	index<N> i0;
	tensor_i<N, double> &t1 = ctrl1.req_block(i0);
	tensor_i<N, double> &t2 = ctrl2.req_block(i0);
	tod_compare<N> op(t1, t2, m_thresh);
	bool res = op.compare();
	m_idx_diff = op.get_diff_index();
	m_diff_elem_1 = op.get_diff_elem_1();
	m_diff_elem_2 = op.get_diff_elem_2();
	return res;
}

template<size_t N>
inline const index<N> &btod_compare<N>::get_diff_index() const {
	return m_idx_diff;
}

template<size_t N>
inline double btod_compare<N>::get_diff_elem_1() const {
	return m_diff_elem_1;
}

template<size_t N>
inline double btod_compare<N>::get_diff_elem_2() const {
	return m_diff_elem_2;
}

} // namespace libtensor

#endif // LIBTENSOR_BTOD_COMPARE_H
