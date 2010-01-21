#ifndef LIBTENSOR_TOD_COMPARE_H
#define LIBTENSOR_TOD_COMPARE_H

#include <cmath>
#include "../defs.h"
#include "../exception.h"
#include "../core/tensor_i.h"
#include "../core/tensor_ctrl.h"

namespace libtensor {

/**	\brief Compares two tensors

	This operation takes two tensors with the same dimensions and
	compares each element. If the difference between the elements
	exceeds a given threshold, the tensors are considered different.

	The class can also return the information about the first found
	difference: the index and the two values.

	Example:
	\code
	tensor_i<2,double> &t1(...), &t2(...);
	tod_compare<2> comp(t1, t2, 1e-16);
	if(!comp.compare()) {
		index<2> idx(comp.get_diff_index());
		double elem1 = comp.get_diff_elem_1();
		double elem2 = comp.get_diff_elem_2();
		cout << "Difference found: "
			<< "[" << idx[0] << "," << idx[1] << "]"
			<< " " << elem1 << " vs " << elem2 << "." << endl;
	} else {
		cout << "No differences found." << endl;
	}
	\endcode

	\ingroup libtensor_tod
**/
template<size_t N>
class tod_compare {
private:
	tensor_i<N,double> &m_t1;
	tensor_i<N,double> &m_t2;
	double m_thresh;
	index<N> m_idx_diff;
	double m_diff_elem_1, m_diff_elem_2;

public:
	/**	\brief Initializes the operation
		\param t1 First %tensor.
		\param t2 Second %tensor.
		\param thresh Threshold.

		The two tensors must have the same dimensions, otherwise an
		exception will be thrown.
	**/
	tod_compare(tensor_i<N,double> &t1, tensor_i<N,double> &t2,
		double thresh) throw(exception);

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
			the first %tensor
	**/
	double get_diff_elem_1() const;

	/**	\brief Returns the value of the first different element in
			the second %tensor
	**/
	double get_diff_elem_2() const;
};

template<size_t N>
inline tod_compare<N>::tod_compare(tensor_i<N,double> &t1,
	tensor_i<N,double> &t2, double thresh) throw(exception) :
	m_t1(t1), m_t2(t2), m_thresh(fabs(thresh)) {
	m_diff_elem_1 = 0.0; m_diff_elem_2 = 0.0;

	const dimensions<N> &dim1(m_t1.get_dims()), &dim2(m_t2.get_dims());
	for(size_t i=0; i<N; i++) if(dim1[i]!=dim2[i]) {
		throw_exc("tod_compare<N>", "tod_compare()",
			"The tensors have different dimensions");
	}

}

template<size_t N>
bool tod_compare<N>::compare() {
	tensor_ctrl<N,double> tctrl1(m_t1), tctrl2(m_t2);
	const double *p1 = tctrl1.req_const_dataptr();
	const double *p2 = tctrl2.req_const_dataptr();

	for(size_t i=0; i<N; i++) m_idx_diff[i]=0;
	size_t sz = m_t1.get_dims().get_size();
	bool equal = true;
	for(size_t i=0; i<sz; i++) {
		if(fabs(p1[i]-p2[i]) > m_thresh) {
			m_diff_elem_1 = p1[i];
			m_diff_elem_2 = p2[i];
			equal = false;
			break;
		}
		m_t1.get_dims().inc_index(m_idx_diff);
	}

	tctrl1.ret_dataptr(p1);
	tctrl2.ret_dataptr(p2);

	return equal;
}

template<size_t N>
inline const index<N> &tod_compare<N>::get_diff_index() const {
	return m_idx_diff;
}

template<size_t N>
inline double tod_compare<N>::get_diff_elem_1() const {
	return m_diff_elem_1;
}

template<size_t N>
inline double tod_compare<N>::get_diff_elem_2() const {
	return m_diff_elem_2;
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_COMPARE_H

