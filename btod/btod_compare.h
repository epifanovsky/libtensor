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
public:
	typedef struct btod_diff {
		bool m_number_of_orbits; //!< same number of orbits
		bool m_similar_orbit; //!< same kind of orbit
		bool m_zero_1, m_zero_2; //!< zero blocks
		//! canonical block indices of orbits in which a difference occured
		index<N> m_canonical_block_index_1, m_canonical_block_index_2;
		index<N> m_inblock; //!< diff index in block
		double m_diff_elem_1, m_diff_elem_2;

		btod_diff() : m_number_of_orbits(true),	m_similar_orbit(true),
			m_zero_1(false), m_zero_2(false),
			m_diff_elem_1(0.0),	m_diff_elem_2(0.0)
		{}

	} btod_diff_t;
private:
	static const char* k_clazz;

	block_tensor_i<N, double> &m_bt1;
	block_tensor_i<N, double> &m_bt2;
	double m_thresh;
	btod_diff_t m_diff_struct;
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

	/**	\brief Returns information about the difference of the two block tensors
	**/
	const btod_diff_t &get_diff() const;

private:
	btod_compare<N> &operator=(const btod_compare<N>&);

};

template<size_t N>
const char* btod_compare<N>::k_clazz="btod_compare<N>";

template<size_t N>
inline btod_compare<N>::btod_compare(block_tensor_i<N, double> &bt1,
	block_tensor_i<N, double> &bt2, double thresh) throw(exception)
	: m_bt1(bt1), m_bt2(bt2), m_thresh(fabs(thresh)) {

	if ( ! m_bt1.get_bis().equals( m_bt2.get_bis() ) )
		throw bad_parameter(g_ns,k_clazz,"btod_compare()",__FILE__,__LINE__,
			"Block index spaces do not match");
}

template<size_t N>
bool btod_compare<N>::compare() {
	block_tensor_ctrl<N, double> ctrl1(m_bt1), ctrl2(m_bt2);

	orbit_list<N,double> orblist1(ctrl1.req_symmetry()),
		orblist2(ctrl2.req_symmetry());

	// if orbit lists are different
	if ( orblist1.get_size() != orblist2.get_size() ) {
		m_diff_struct.m_number_of_orbits=false;
		return false;
	}

	// run over all orbits
	for ( typename orbit_list<N,double>::iterator it=orblist1.begin();
		  it!=orblist1.end(); it++ ) {
		orbit<N,double> orb1(ctrl1.req_symmetry(),it->second);
		orbit<N,double> orb2(ctrl2.req_symmetry(),it->second);

		size_t idx1=orb1.get_abs_canonical_index(),
			idx2=orb2.get_abs_canonical_index();

		index<N> bidx1, bidx2;
		m_bt1.get_bis().get_block_index_dims().abs_index(idx1,bidx1);
		m_bt2.get_bis().get_block_index_dims().abs_index(idx2,bidx2);

		// if number of blocks in the orbits is different
		if ( orb1.get_size() != orb1.get_size() ) {
			m_diff_struct.m_similar_orbit=false;
			m_diff_struct.m_canonical_block_index_1=bidx1;
			m_diff_struct.m_canonical_block_index_2=bidx2;
			return false;
		}

		if ( idx1 != idx2 ) {
			// tra: transfrom from canonical block of orb1 into the canonical block
			// of orb2
			transf<N, double> tra;
			tra.transform(orb1.get_transf(idx1));

			// run over all blocks in orbit and check whether they are the same
			for ( typename orbit<N,double>::iterator itorb=orb1.begin();
					itorb!=orb1.end(); itorb++ ) {
				transf<N, double> trb(tra);
				trb.transform(orb2.get_transf(orb1.get_abs_index(itorb)));
				if ( trb != orb1.get_transf(itorb) ) {
					m_diff_struct.m_similar_orbit=false;
					m_diff_struct.m_canonical_block_index_1=bidx1;
					m_diff_struct.m_canonical_block_index_2=bidx2;
					return false;
				}
			}

			bool zero_1 = ctrl1.req_is_zero_block(bidx1);
			bool zero_2 = ctrl2.req_is_zero_block(bidx2);

			if(zero_1 != zero_2) {
				m_diff_struct.m_zero_1 = zero_1;
				m_diff_struct.m_zero_2 = zero_2;
				m_diff_struct.m_canonical_block_index_1=bidx1;
				m_diff_struct.m_canonical_block_index_2=bidx2;

				return false;
			}

			if(!zero_1) {

				tensor_i<N,double> &t1=ctrl1.req_block(bidx1),
					&t2=ctrl2.req_block(bidx2);

				dimensions<N> d1=t1.get_dims(), d2=t2.get_dims();
				d1.permute(tra.get_perm());

				tensor<N,double,libvmm::std_allocator<double> > tmp(d1);
				tod_copy<N> docopy(t1,tra.get_perm(),tra.get_coeff());
				docopy.perform(tmp);

				tod_compare<N> compare(tmp,t2,m_thresh);
				if ( ! compare.compare() ) {
					m_diff_struct.m_canonical_block_index_1=bidx1;
					m_diff_struct.m_canonical_block_index_2=bidx2;
					m_diff_struct.m_inblock=compare.get_diff_index();
					m_diff_struct.m_diff_elem_1=compare.get_diff_elem_1();
					m_diff_struct.m_diff_elem_2=compare.get_diff_elem_2();

					return false;
				}
			}
		}
		else {
			// run over all blocks in orbit and check whether they are the same
			for ( typename orbit<N,double>::iterator itorb=orb1.begin();
					itorb!=orb1.end(); itorb++ ) {
				if ( orb1.get_transf(itorb) !=
					orb2.get_transf(orb1.get_abs_index(itorb) ) ) {
					m_diff_struct.m_similar_orbit=false;
					m_diff_struct.m_canonical_block_index_1=bidx1;
					m_diff_struct.m_canonical_block_index_2=bidx2;
					return false;
				}
			}

			bool zero_1 = ctrl1.req_is_zero_block(bidx1);
			bool zero_2 = ctrl2.req_is_zero_block(bidx2);

			if(zero_1 != zero_2) {
				m_diff_struct.m_zero_1 = zero_1;
				m_diff_struct.m_zero_2 = zero_2;
				m_diff_struct.m_canonical_block_index_1=bidx1;
				m_diff_struct.m_canonical_block_index_2=bidx2;

				return false;
			}

			if(!zero_1) {

				tensor_i<N,double> &t1=ctrl1.req_block(bidx1),
					&t2=ctrl2.req_block(bidx2);

				tod_compare<N> compare(t1,t2,m_thresh);
				if ( ! compare.compare() ) {
					m_diff_struct.m_canonical_block_index_1=bidx1;
					m_diff_struct.m_canonical_block_index_2=bidx2;
					m_diff_struct.m_inblock=compare.get_diff_index();
					m_diff_struct.m_diff_elem_1=compare.get_diff_elem_1();
					m_diff_struct.m_diff_elem_2=compare.get_diff_elem_2();

					return false;
				}

				ctrl1.ret_block(bidx1);
				ctrl2.ret_block(bidx2);
			}
		}
	}

	return true;
}

template<size_t N>
inline const typename btod_compare<N>::btod_diff_t &btod_compare<N>::get_diff()
	const {

	return m_diff_struct;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_COMPARE_H
