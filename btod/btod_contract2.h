#ifndef LIBTENSOR_BTOD_CONTRACT2_H
#define LIBTENSOR_BTOD_CONTRACT2_H

#include "defs.h"
#include "exception.h"
#include "core/block_tensor_i.h"
#include "core/block_tensor_ctrl.h"
#include "tod/contraction2.h"
#include "tod/tod_contract2.h"
#include "btod/btod_additive.h"
#include "symmetry/so_projdown.h"
#include "symmetry/so_projup.h"

namespace libtensor {


/**	\brief Contraction of two block tensors

	\ingroup libtensor_btod
 **/
template<size_t N, size_t M, size_t K>
class btod_contract2 : public btod_additive<N + M> {
public:
	static const char *k_clazz; //!< Class name

private:
	static const size_t k_ordera = N + K; //!< Order of first argument (a)
	static const size_t k_orderb = M + K; //!< Order of second argument (b)
	static const size_t k_orderc = N + M; //!< Order of result (c)
	static const size_t k_totidx = N + M + K; //!< Total number of indexes
	static const size_t k_maxconn = 2 * k_totidx; //!< Index connections

private:
	contraction2<N, M, K> m_contr; //!< Contraction
	block_tensor_i<k_ordera, double> &m_bta; //!< First argument (a)
	block_tensor_i<k_orderb, double> &m_btb; //!< Second argument (b)
	block_index_space<k_orderc> m_bis; //!< Block %index space of the result
	symmetry<k_orderc, double> m_sym; //!< Symmetry of the result

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the contraction operation
		\param contr Contraction.
		\param bta Block %tensor a (first argument).
		\param btb Block %tensor b (second argument).
	 **/
	btod_contract2(const contraction2<N, M, K> &contr,
		block_tensor_i<k_ordera, double> &bta,
		block_tensor_i<k_orderb, double> &btb);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_contract2();

	//@}

	//!	\name Implementation of libtensor::btod_additive<N + M>
	//@{
	virtual void perform(block_tensor_i<k_orderc, double> &btc, double c)
		throw(exception);
	//@}

	//!	\name Implementation of
	//		libtensor::direct_block_tensor_operation<N + M, double>
	//@{
	virtual const block_index_space<N + M> &get_bis() const;
	virtual const symmetry<N + M, double> &get_symmetry() const;
	virtual void perform(block_tensor_i<k_orderc, double> &btc)
		throw(exception);
	//@}

private:
	static block_index_space<N + M> make_bis(
		const contraction2<N, M, K> &contr,
		block_tensor_i<k_ordera, double> &bta,
		block_tensor_i<k_orderb, double> &btb);
	void make_symmetry();

	/**	\brief For an orbit in a and b, make a list of blocks in c
	 **/
	void make_list_2orbits(const orbit<k_ordera, double> &orba,
		const orbit<k_orderb, double> &orbb);
};


template<size_t N, size_t M, size_t K>
const char *btod_contract2<N, M, K>::k_clazz = "btod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
btod_contract2<N, M, K>::btod_contract2(const contraction2<N, M, K> &contr,
	block_tensor_i<k_ordera, double> &bta,
	block_tensor_i<k_orderb, double> &btb)
: m_contr(contr), m_bta(bta), m_btb(btb), m_bis(make_bis(contr, bta, btb)),
	m_sym(m_bis) {

	make_symmetry();
}


template<size_t N, size_t M, size_t K>
btod_contract2<N, M, K>::~btod_contract2() {

}


template<size_t N, size_t M, size_t K>
inline const block_index_space<N+M> &btod_contract2<N, M, K>::get_bis() const {

	return m_bis;
}


template<size_t N, size_t M, size_t K>
const symmetry<N + M, double> &btod_contract2<N, M, K>::get_symmetry() const {

	return m_sym;
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(block_tensor_i<k_orderc, double> &btc,
	double c) throw(exception) {

	index<k_ordera> idx_a;
	index<k_orderb> idx_b;
	index<k_orderc> idx_c;

	block_tensor_ctrl<k_orderc, double> ctrl_btc(btc);
	block_tensor_ctrl<k_ordera, double> ctrl_bta(m_bta);
	block_tensor_ctrl<k_orderb, double> ctrl_btb(m_btb);

	tod_contract2<N,M,K> contr(m_contr,ctrl_bta.req_block(idx_a),ctrl_btb.req_block(idx_b));

	contr.perform(ctrl_btc.req_block(idx_c),c);

	dimensions<k_ordera> bidimsa(m_bta.get_bis().get_block_index_dims());
	dimensions<k_orderb> bidimsb(m_btb.get_bis().get_block_index_dims());

	orbit_list<k_ordera, double> orblsta(ctrl_bta.req_symmetry());
	orbit_list<k_orderb, double> orblstb(ctrl_btb.req_symmetry());
	typename orbit_list<k_ordera, double>::iterator iorba = orblsta.begin();
	for(; iorba != orblsta.end(); iorba++) {
		orbit<k_ordera, double> orba(ctrl_bta.req_symmetry(), *iorba);
		typename orbit_list<k_orderb, double>::iterator iorbb =
			orblstb.begin();
		for(; iorbb != orblstb.end(); iorbb++) {
			orbit<k_orderb, double> orbb(
				ctrl_btb.req_symmetry(), *iorbb);
			make_list_2orbits(orba, orbb);
		}
	}
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::perform(block_tensor_i<k_orderc, double> &btc)
	throw(exception) {

	index<k_ordera> idx_a;
	index<k_orderb> idx_b;
	index<k_orderc> idx_c;

	block_tensor_ctrl<k_orderc, double> ctrl_btc(btc);
	block_tensor_ctrl<k_ordera, double> ctrl_bta(m_bta);
	block_tensor_ctrl<k_orderb, double> ctrl_btb(m_btb);

	tod_contract2<N,M,K> contr(m_contr,ctrl_bta.req_block(idx_a),ctrl_btb.req_block(idx_b));

	contr.perform(ctrl_btc.req_block(idx_c));
}


template<size_t N, size_t M, size_t K>
block_index_space<N + M> btod_contract2<N, M, K>::make_bis(
	const contraction2<N, M, K> &contr,
	block_tensor_i<k_ordera, double> &bta,
	block_tensor_i<k_orderb, double> &btb) {

	static const char *method = "make_bis()";

	const block_index_space<k_ordera> &bisa = bta.get_bis();
	const block_index_space<k_orderb> &bisb = btb.get_bis();

	//	Check if contracted indexes are compatible

	const sequence<k_maxconn, size_t> &conn = contr.get_conn();
	for(size_t idima = 0; idima < k_ordera; idima++) {
		size_t iconn = conn[k_orderc + idima];
		if(iconn >= k_orderc + k_ordera) {
			size_t idimb = conn[k_orderc + idima] -
				k_orderc - k_ordera;
			size_t itypa = bisa.get_type(idima);
			size_t itypb = bisb.get_type(idimb);
			if(!bisa.get_splits(itypa).equals(
				bisb.get_splits(itypb))) {
				throw bad_parameter(g_ns, k_clazz, method,
					__FILE__, __LINE__,
					"Block tensor dimensions are unsuitable"
					" for contraction.");
			}
		}
	}

	//	Build the result block index space

	index<k_orderc> i0, i1;
	for(size_t idimc = 0; idimc < k_orderc; idimc++) {
		size_t iconn = conn[idimc];
		if(iconn >= k_orderc + k_ordera) {
			const dimensions<k_orderb> &dims = bisb.get_dims();
			i1[idimc] = dims[iconn - k_orderc - k_ordera] - 1;
		} else {
			const dimensions<k_ordera> &dims = bisa.get_dims();
			i1[idimc] = dims[iconn - k_orderc] - 1;
		}
	}
	block_index_space<k_orderc> bis(dimensions<k_orderc>(
		index_range<k_orderc>(i0, i1)));
	mask<k_orderc> msk_done;
	for(size_t idimc = 0; idimc < k_orderc; idimc++) {
		if(msk_done[idimc]) continue;
		mask<k_orderc> msk_todo;
		if(conn[idimc] >= k_orderc + k_ordera) {
			size_t type = bisb.get_type(
				conn[idimc] - k_orderc - k_ordera);
			for(size_t idimb = 0; idimb < k_orderb; idimb++) {
				size_t iconn = k_orderc + k_ordera + idimb;
				if(bisb.get_type(idimb) == type &&
					conn[iconn] < k_orderc) {
					msk_todo[conn[iconn]] = true;
				}
			}
			const split_points &pts = bisb.get_splits(type);
			size_t npts = pts.get_num_points();
			for(size_t ipt = 0; ipt < npts; ipt++)
				bis.split(msk_todo, pts[ipt]);
		} else {
			size_t type = bisa.get_type(conn[idimc] - k_orderc);
			for(size_t idima = 0; idima < k_ordera; idima++) {
				size_t iconn = k_orderc + idima;
				if(bisa.get_type(idima) == type &&
					conn[iconn] < k_orderc) {
					msk_todo[conn[iconn]] = true;
				}
			}
			const split_points &pts = bisa.get_splits(type);
			size_t npts = pts.get_num_points();
			for(size_t ipt = 0; ipt < npts; ipt++)
				bis.split(msk_todo, pts[ipt]);
		}
		msk_done |= msk_todo;
	}

	return bis;
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_symmetry() {

	const sequence<k_maxconn, size_t> &conn = m_contr.get_conn();

	dimensions<k_ordera> bidimsa(m_bta.get_bis().get_block_index_dims());
	dimensions<k_orderb> bidimsb(m_btb.get_bis().get_block_index_dims());
	dimensions<k_orderc> bidimsc(m_bis.get_block_index_dims());

	index<N> ia1, ia2;
	mask<k_ordera> projmska;
	mask<k_orderc> projmskca;
	size_t j = 0;
	for(size_t i = 0; i < k_ordera; i++) {
		size_t iconn = conn[k_orderc + i];
		if(iconn < k_orderc) {
			ia2[j] = bidimsa[i] - 1;
			projmska[i] = true;
			projmskca[iconn] = true;
		}
	}
	dimensions<N> projdimsa(index_range<N>(ia1, ia2));
	block_tensor_ctrl<k_ordera, double> ctrla(m_bta);
	const symmetry<k_ordera, double> &syma = ctrla.req_symmetry();
	size_t nelem = syma.get_num_elements();
	for(size_t ielem = 0; ielem < nelem; ielem++) {
		so_projdown<k_ordera, K, double> projdn(
			syma.get_element(ielem), projmska, projdimsa);
		if(!projdn.is_identity()) {
			so_projup<N, M, double> projup(
				projdn.get_proj(), projmskca, bidimsc);
			m_sym.add_element(projup.get_proj());
		}
	}

	index<M> ib1, ib2;
	mask<k_orderb> projmskb;
	mask<k_orderc> projmskcb;
	j = 0;
	for(size_t i = 0; i < k_orderb; i++) {
		size_t iconn = conn[k_orderc + k_ordera + i];
		if(iconn < k_orderc) {
			ib2[j] = bidimsb[i] - 1;
			projmskb[i] = true;
			projmskcb[iconn] = true;
		}
	}
	dimensions<M> projdimsb(index_range<M>(ib1, ib2));
	block_tensor_ctrl<k_orderb, double> ctrlb(m_btb);
	const symmetry<k_orderb, double> &symb = ctrlb.req_symmetry();
	nelem = symb.get_num_elements();
	for(size_t ielem = 0; ielem < nelem; ielem++) {
		so_projdown<k_orderb, K, double> projdn(
			symb.get_element(ielem), projmskb, projdimsb);
		if(!projdn.is_identity()) {
			so_projup<M, N, double> projup(
				projdn.get_proj(), projmskcb, bidimsc);
			m_sym.add_element(projup.get_proj());
		}
	}
}


template<size_t N, size_t M, size_t K>
void btod_contract2<N, M, K>::make_list_2orbits(
	const orbit<k_ordera, double> &orba,
	const orbit<k_orderb, double> &orbb) {

	index<k_ordera> idxa;
	index<k_orderb> idxb;
	index<k_orderc> idxc;

	const sequence<k_maxconn, size_t> &conn = m_contr.get_conn();
	for(size_t i = 0; i < k_ordera; i++) {
		if(conn[k_orderc + i] < k_orderc)
			idxc[conn[k_orderc + i]] = idxa[i];
	}
	for(size_t i = 0; i < k_orderb; i++) {
		if(conn[k_orderc + k_ordera + i] < k_orderc)
			idxc[conn[k_orderc + k_ordera + i]] = idxb[i];
	}

}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_H
