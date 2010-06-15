#ifndef LIBTENSOR_BTOD_DIAG_H
#define LIBTENSOR_BTOD_DIAG_H

#include "../defs.h"
#include "../not_implemented.h"
#include "../core/abs_index.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../core/permutation_builder.h"
#include "../symmetry/so_copy.h"
#include "../symmetry/so_add.h"
#include "../symmetry/so_permute.h"
#include "../symmetry/so_proj_down.h"
#include "../tod/tod_copy.h"
#include "../tod/tod_diag.h"
#include "../tod/tod_set.h"
#include "bad_block_index_space.h"
#include "additive_btod.h"
#include "transf_double.h"

namespace libtensor {


/**	\brief Extracts a general diagonal from a block %tensor
	\tparam N Tensor order.
	\tparam M Diagonal order.

	\ingroup libtensor_btod
 **/
template<size_t N, size_t M>
class btod_diag :
	public additive_btod<N - M + 1>,
	public timings< btod_diag<N, M> > {

public:
	static const char *k_clazz; //!< Class name

public:
	static const size_t k_ordera = N; //!< Order of the argument
	static const size_t k_orderb = N - M + 1; //!< Order of the result

private:
	block_tensor_i<N, double> &m_bta; //!< Input block %tensor
	mask<N> m_msk; //!< Diagonal %mask
	permutation<k_orderb> m_perm; //!< Permutation of the result
	double m_c; //!< Scaling coefficient
	block_index_space<k_orderb> m_bis; //!< Block %index space of the result
	symmetry<k_orderb, double> m_sym; //!< Symmetry of the result
	assignment_schedule<k_orderb, double> m_sch; //!< Assignment schedule

public:
	//!	\name Construction and destruction
	//@{

	/** \brief Initializes the diagonal extraction operation
		\param bta Input block %tensor
		\param msk Mask which specifies the indexes to take the diagonal
		\param c Scaling factor
	 **/
	btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
		double c = 1.0);

	/** \brief Initializes the diagonal extraction operation
		\param bta Input block %tensor
		\param msk Mask which specifies the indexes to take the diagonal
		\param p Permutation of result tensor
		\param c Scaling factor
	 **/
	btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
		const permutation<N - M + 1> &p, double c = 1.0);

	//@}

	//!	\name Implementation of
	//		libtensor::direct_tensor_operation<N - M + 1, double>
	//@{

	virtual const block_index_space<k_orderb> &get_bis() const {
		return m_bis;
	}

	virtual const symmetry<k_orderb, double> &get_symmetry() const {
		return m_sym;
	}

	virtual const assignment_schedule<k_orderb, double> &get_schedule() const {
		return m_sch;
	}

	//@}

	using additive_btod<k_orderb>::perform;

protected:
	virtual void compute_block(tensor_i<k_orderb, double> &blk,
		const index<k_orderb> &i);

	virtual void compute_block(tensor_i<k_orderb, double> &blk,
		const index<k_orderb> &i, const transf<k_orderb, double> &tr, double c);

private:
	/**	\brief Forms the block %index space of the output or throws an
			exception if the input is incorrect.
	 **/
	static block_index_space<N - M + 1> mk_bis(
		const block_index_space<N> &bis, const mask<N> &msk);

	/**	\brief Sets up the assignment schedule for the operation.
	 **/
	void make_schedule();

private:
	btod_diag(const btod_diag<N, M>&);
	const btod_diag<N, M> &operator=(const btod_diag<N, M>&);

};


template<size_t N, size_t M>
const char *btod_diag<N, M>::k_clazz = "btod_diag<N, M>";


template<size_t N, size_t M>
btod_diag<N, M>::btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
	double c) :

	m_bta(bta), m_msk(m), m_c(c),
	m_bis(mk_bis(bta.get_bis(), m_msk)),
	m_sym(m_bis), m_sch(m_bis.get_block_index_dims()) {

	mask<N> msk;
	bool not_done = true;
	for (size_t i = 0; i < N; i++) {
		if (! m[i] ) msk[i] = true;
		else if ( not_done ) {
			msk[i] = true;
			not_done = false;
		}
	}
	block_tensor_ctrl<N, double> ctrla(bta);
	so_proj_down<N, M - 1, double>(ctrla.req_const_symmetry(), msk).perform(m_sym);

	make_schedule();
}


template<size_t N, size_t M>
btod_diag<N, M>::btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
	const permutation<N - M + 1> &p, double c) :

	m_bta(bta), m_msk(m), m_perm(p), m_c(c),
	m_bis(mk_bis(bta.get_bis(), m_msk)),
	m_sym(m_bis), m_sch(m_bis.get_block_index_dims())  {

	mask<N> msk;
	bool not_done = true;
	for (size_t i = 0; i < N; i++) {
		if (! m[i] ) msk[i] = true;
		else if ( not_done ) {
			msk[i] = true;
			not_done = false;
		}
	}
	symmetry<N - M + 1, double> sym1(m_bis);
	block_tensor_ctrl<N, double> ctrla(bta);
	so_proj_down<N, M - 1, double>(ctrla.req_const_symmetry(), msk).perform(sym1);
	so_permute<N - M + 1, double>(sym1, p).perform(m_sym);
	m_bis.permute(p);

	make_schedule();
}


template<size_t N, size_t M>
void btod_diag<N, M>::compute_block(tensor_i<k_orderb, double> &blk,
	const index<k_orderb> &idx) {

	btod_diag<N, M>::start_timer();

	block_tensor_ctrl<N, double> ctrla(m_bta);
	dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

	permutation<k_orderb> pinv(m_perm, true);
	size_t map[k_ordera];
	size_t j = 0, jd;
	bool b = false;
	for(size_t i = 0; i < k_ordera; i++) {
		if(m_msk[i]) {
			if(b) map[i] = jd;
			else { map[i] = jd = j++; b = true; }
		} else {
			map[i] = j++;
		}
	}

	index<k_ordera> idxa;
	index<k_orderb> idxb(idx);
	idxb.permute(pinv);
	for(size_t i = 0; i < k_ordera; i++) idxa[i] = idxb[map[i]];

	orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), idxa);
	abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(), bidimsa);
	const transf<k_ordera, double> &tra = oa.get_transf(idxa);

	// Extract diagonal of block of bta into block of btb
	//
	size_t seqa1[k_ordera], seqa2[k_ordera];
	size_t seqb1[k_orderb], seqb2[k_orderb];
	for(register size_t i = 0; i < k_ordera; i++)
		seqa2[i] = seqa1[i] = i;
	tra.get_perm().apply(seqa2);
	for(register size_t i = 0; i < k_ordera; i++) {
		seqb1[map[i]] = seqa1[i];
		seqb2[map[i]] = seqa2[i];
	}
	permutation_builder<k_orderb> pb(seqb2, seqb1);

	tensor_i<k_ordera, double> &blka = ctrla.req_block(cidxa.get_index());

	permutation<k_orderb> permb(pb.get_perm());
	permb.permute(m_perm);

	tod_diag<N, M>(blka, m_msk, permb, m_c * tra.get_coeff()).perform(blk);

	ctrla.ret_block(cidxa.get_index());

	btod_diag<N, M>::stop_timer();

}

template<size_t N, size_t M>
void btod_diag<N, M>::compute_block(tensor_i<k_orderb, double> &blk,
	const index<k_orderb> &idx, const transf<k_orderb, double> &tr, double c) {

	btod_diag<N, M>::start_timer();

	block_tensor_ctrl<N, double> ctrla(m_bta);
	dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

	permutation<k_orderb> pinv(m_perm, true);
	size_t map[k_ordera];
	size_t j = 0, jd;
	bool b = false;
	for(size_t i = 0; i < k_ordera; i++) {
		if(m_msk[i]) {
			if(b) map[i] = jd;
			else { map[i] = jd = j++; b = true; }
		} else {
			map[i] = j++;
		}
	}

	index<k_ordera> idxa;
	index<k_orderb> idxb(idx);
	idxb.permute(pinv);
	for(size_t i = 0; i < k_ordera; i++) idxa[i] = idxb[map[i]];

	orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), idxa);
	abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(), bidimsa);
	const transf<k_ordera, double> &tra = oa.get_transf(idxa);

	// Extract diagonal of block of bta into block of btb
	//
	size_t seqa1[k_ordera], seqa2[k_ordera];
	size_t seqb1[k_orderb], seqb2[k_orderb];
	for(register size_t i = 0; i < k_ordera; i++)
		seqa2[i] = seqa1[i] = i;
	tra.get_perm().apply(seqa2);
	for(register size_t i = 0; i < k_ordera; i++) {
		seqb1[map[i]] = seqa1[i];
		seqb2[map[i]] = seqa2[i];
	}
	permutation_builder<k_orderb> pb(seqb2, seqb1);

	tensor_i<k_ordera, double> &blka = ctrla.req_block(cidxa.get_index());

	permutation<k_orderb> permb(pb.get_perm());
	permb.permute(m_perm);
	permb.permute(tr.get_perm());

	tod_diag<N, M>(blka, m_msk, permb,
			m_c * tra.get_coeff() * tr.get_coeff()).perform(blk, c);

	ctrla.ret_block(cidxa.get_index());

	btod_diag<N, M>::stop_timer();
}


template<size_t N, size_t M>
block_index_space<N - M + 1> btod_diag<N, M>::mk_bis(
	const block_index_space<N> &bis, const mask<N> &msk) {

	static const char *method =
		"mk_bis(const block_index_space<N>&, const mask<N>&)";

	//	Verify identical types on the block index space diagonal
	//
	size_t typ;
	bool typ_defined = false;
	for(size_t i = 0; i < N; i++) {
		if(!msk[i]) continue;
		if(!typ_defined) {
			typ = bis.get_type(i);
			typ_defined = true;
		} else {
			if(bis.get_type(i) != typ) {
				throw bad_block_index_space(g_ns, k_clazz,
					method, __FILE__, __LINE__, "bt");
			}
		}
	}

	//	Input dimensions
	dimensions<N> idims(bis.get_dims());

	//	Compute output dimensions
	//
	index<k_orderb> i1, i2;

	size_t m = 0, j = 0;
	size_t d = 0;
	size_t map[k_orderb];
	for(size_t i = 0; i < N; i++) {
		if(msk[i]) {
			m++;
			if(d == 0) {
				d = idims[i];
				i2[j] = d - 1;
				map[j] = i;
				j++;
			}
		} else {
			i2[j] = idims[i] - 1;
			map[j] = i;
			j++;
		}
	}
	if(m != M) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"m");
	}

	//	Output block index space
	block_index_space<k_orderb> obis(dimensions<k_orderb>(
		index_range<k_orderb>(i1, i2)));

	mask<k_orderb> msk_done;
	bool done = false;
	while(!done) {
		size_t i = 0;
		while(i < k_orderb && msk_done[i]) i++;
		if(i == k_orderb) {
			done = true;
			continue;
		}
		size_t typ = bis.get_type(map[i]);
		const split_points &splits = bis.get_splits(typ);
		mask<k_orderb> msk_typ;
		for(size_t j = 0; j < k_orderb; j++) {
			if(bis.get_type(map[j]) == typ) msk_typ[j] = true;
		}
		size_t npts = splits.get_num_points();
		for(register size_t j = 0; j < npts; j++) {
			obis.split(msk_typ, splits[j]);
		}
		msk_done |= msk_typ;
	}
	obis.match_splits();

	return obis;
}

template<size_t N, size_t M>
void btod_diag<N, M>::make_schedule() {

	block_tensor_ctrl<N, double> ctrla(m_bta);
	dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

	permutation<k_orderb> pinv(m_perm, true);
	size_t map[k_ordera];
	size_t j = 0, jd;
	bool b = false;
	for(size_t i = 0; i < k_ordera; i++) {
		if(m_msk[i]) {
			if(b) map[i] = jd;
			else { map[i] = jd = j++; b = true; }
		} else {
			map[i] = j++;
		}
	}

	orbit_list<k_orderb, double> olb(m_sym);
	for (typename orbit_list<k_orderb, double>::iterator iob = olb.begin();
			iob != olb.end(); iob++) {

		index<k_ordera> idxa;
		index<k_orderb> idxb(olb.get_index(iob));

		for(size_t i = 0; i < k_ordera; i++) idxa[i] = idxb[map[i]];

		orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), idxa);
		abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(), bidimsa);

		if(ctrla.req_is_zero_block(cidxa.get_index())) continue;

		m_sch.insert(olb.get_abs_index(iob));
	}
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_DIAG_H
