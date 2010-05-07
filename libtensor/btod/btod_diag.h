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
public:
	btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
		double c = 1.0);

	btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
		const permutation<N - M + 1> &p, double c = 1.0);

	virtual const block_index_space<k_orderb> &get_bis() const {
		return m_bis;
	}

	virtual const symmetry<k_orderb, double> &get_symmetry() const {
		return m_sym;
	}

	virtual const assignment_schedule<N - M + 1, double> &get_schedule() const;

	virtual void compute_block(tensor_i<k_orderb, double> &blk,
		const index<k_orderb> &i) { }
	virtual void compute_block(tensor_i<k_orderb, double> &blk,
		const index<k_orderb> &i, const transf<k_orderb, double> &tr, double c) { }

	virtual void perform(block_tensor_i<k_orderb, double> &btb)
		throw(exception);

	virtual void perform(block_tensor_i<k_orderb, double> &btb, double c)
		throw(exception);

	virtual void perform(block_tensor_i<k_orderb, double> &btb,
		const index<k_orderb> &idx) throw(exception) {

		throw not_implemented(g_ns, k_clazz, "perform()",
			__FILE__, __LINE__);
	}

	virtual void perform(block_tensor_i<k_orderb, double> &btb,
		const index<k_orderb> &idx, double c) throw(exception) {

		throw not_implemented(g_ns, k_clazz, "perform()",
			__FILE__, __LINE__);
	}

private:
	/**	\brief Forms the block %index space of the output or throws an
			exception if the input is incorrect
	 **/
	static block_index_space<N - M + 1> mk_bis(
		const block_index_space<N> &bis, const mask<N> &msk);

	void do_perform_zero(block_tensor_i<k_orderb, double> &btb, double c)
		throw(exception);
	void do_perform_nozero(block_tensor_i<k_orderb, double> &btb, double c)
		throw(exception);
	void do_perform(block_tensor_i<k_orderb, double> &btb,
			const index<k_orderb> &dst_idx, bool zero, double c)
		throw(exception);
private:
	btod_diag(const btod_diag<N, M>&);
	const btod_diag<N, M> &operator=(const btod_diag<N, M>&);

};


template<size_t N, size_t M>
const char *btod_diag<N, M>::k_clazz = "btod_diag<N, M>";


template<size_t N, size_t M>
btod_diag<N, M>::btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
	double c) :

	m_bta(bta), m_msk(m), m_c(c), m_bis(mk_bis(bta.get_bis(), m_msk)),
	m_sym(m_bis) {

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

}


template<size_t N, size_t M>
btod_diag<N, M>::btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
	const permutation<N - M + 1> &p, double c) :

	m_bta(bta), m_msk(m), m_perm(p), m_c(c),
	m_bis(mk_bis(bta.get_bis(), m_msk)), m_sym(m_bis) {

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

}


template<size_t N, size_t M>
const assignment_schedule<N - M + 1, double> &btod_diag<N, M>::get_schedule() const {

	throw not_implemented(g_ns, k_clazz, "get_schedule()",
		__FILE__, __LINE__);
}


template<size_t N, size_t M>
void btod_diag<N, M>::perform(block_tensor_i<k_orderb, double> &btb)
	throw(exception) {

	static const char *method =
		"perform(block_tensor_i<N - M + 1, double>&)";

	if(!btb.get_bis().equals(m_bis)) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "btb");
	}

	do_perform_zero(btb, 1.0);
}


template<size_t N, size_t M>
void btod_diag<N, M>::perform(block_tensor_i<k_orderb, double> &btb, double c)
	throw(exception) {

	static const char *method =
		"perform(block_tensor_i<N - M + 1, double>&, double)";

	if(!btb.get_bis().equals(m_bis)) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "btb");
	}

	do_perform_nozero(btb, c);
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
void btod_diag<N, M>::do_perform_nozero(
	block_tensor_i<k_orderb, double> &btb, double c)
	throw (exception) {

	btod_diag<N, M>::start_timer();

	block_tensor_ctrl<k_ordera, double> ctrla(m_bta);
	block_tensor_ctrl<k_orderb, double> ctrlb(btb);

	dimensions<k_ordera> bidimsa = m_bta.get_bis().get_block_index_dims();
	dimensions<k_orderb> bidimsb = m_bis.get_block_index_dims();

	//	Retain a copy of old symmetry in B
	//
	symmetry<k_orderb, double> symb_old(m_bis);
	so_copy<k_orderb, double>(ctrlb.req_const_symmetry()).perform(symb_old);

	//	Install a new symmetry in B
	//
	permutation<k_orderb> p0;
	so_add<k_orderb, double>(m_sym, p0,
			symb_old, p0).perform(ctrlb.req_symmetry());

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

	orbit_list<k_orderb, double> olb1(symb_old),
			olb2(ctrlb.req_const_symmetry());


	//	First go over blocks in b which got "downgraded",
	//	that is turned unique from being replicas
	//
	for(typename orbit_list<k_orderb, double>::iterator iob = olb2.begin();
			iob != olb2.end(); iob++) {

		if (olb1.contains(olb2.get_abs_index(iob))) continue;

		index<k_ordera> idxa;
		index<k_orderb> idxb(olb2.get_index(iob));

		idxb.permute(pinv);
		for(size_t i = 0; i < k_ordera; i++) idxa[i] = idxb[map[i]];
		idxb.permute(m_perm);

		orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), idxa);
		abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(), bidimsa);
		const transf<k_ordera, double> &tra = oa.get_transf(idxa);

		orbit<k_orderb, double> ob(symb_old, idxb);
		abs_index<k_orderb> cidxb(ob.get_abs_canonical_index(), bidimsb);

		bool zeroa = ctrla.req_is_zero_block(cidxa.get_index());
		bool zerob = ctrlb.req_is_zero_block(cidxb.get_index());

		if (zeroa && zerob) {
			ctrlb.req_zero_block(idxb);
			continue;
		}

		tensor_i<k_orderb, double> &blkb = ctrlb.req_block(idxb);

		// Copy previously canonical block onto "downgraded" block
		//
		{
			tensor_i<k_orderb, double> &blkb_can =
					ctrlb.req_block(cidxb.get_index());

			const transf<k_orderb, double> &trb = ob.get_transf(idxb);
			tod_copy<k_orderb>(blkb_can,
					trb.get_perm(), trb.get_coeff()).perform(blkb);

			ctrlb.ret_block(cidxb.get_index());
		}

		// If block in bta is zero, we are done.
		//
		if (zeroa) {
			ctrlb.ret_block(idxb);
			continue;
		}

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

		tensor_i<k_ordera, double> &blka =
				ctrla.req_block(cidxa.get_index());

		permutation<k_orderb> permb(pb.get_perm());
		permb.permute(m_perm);

		tod_diag<N, M>(blka, m_msk, permb, tra.get_coeff()).
				perform(blkb, c);

		ctrla.ret_block(cidxa.get_index());
		ctrlb.ret_block(idxb);

	}

	//	Go over blocks in B that stay canonical
	//
	for(typename orbit_list<k_orderb, double>::iterator iob = olb2.begin();
		iob != olb2.end(); iob++) {

		if(!olb1.contains(olb2.get_abs_index(iob))) continue;

		index<k_ordera> idxa;
		index<k_orderb> idxb(olb2.get_index(iob));
		idxb.permute(pinv);

		for(size_t i = 0; i < k_ordera; i++) idxa[i] = idxb[map[i]];
		orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), idxa);
		abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(), bidimsa);
		const transf<k_ordera, double> &tra = oa.get_transf(idxa);

		bool zeroa = ctrla.req_is_zero_block(cidxa.get_index());
		bool zerob = ctrlb.req_is_zero_block(idxb);

		if (zeroa) {
			continue;
		}

		tensor_i<k_orderb, double> &blkb = ctrlb.req_block(idxb);

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

		tensor_i<k_ordera, double> &blka =
				ctrla.req_block(cidxa.get_index());

		permutation<k_orderb> permb(pb.get_perm());
		permb.permute(m_perm);

		tod_diag<N, M>(blka, m_msk, permb, tra.get_coeff()).
				perform(blkb, c);

		ctrla.ret_block(cidxa.get_index());
		ctrlb.ret_block(idxb);

	}

	btod_diag<N, M>::stop_timer();
}

template<size_t N, size_t M>
void btod_diag<N, M>::do_perform_zero(
	block_tensor_i<k_orderb, double> &btb, double c)
	throw (exception) {

	btod_diag<N, M>::start_timer();

	block_tensor_ctrl<k_ordera, double> ctrla(m_bta);
	block_tensor_ctrl<k_orderb, double> ctrlb(btb);

	dimensions<k_ordera> bidimsa = m_bta.get_bis().get_block_index_dims();
	dimensions<k_orderb> bidimsb = m_bis.get_block_index_dims();

	//	Install a new symmetry in B
	//
	so_copy<k_orderb, double>(m_sym).perform(ctrlb.req_symmetry());

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

	orbit_list<k_orderb, double> olb(ctrlb.req_const_symmetry());

	for(typename orbit_list<k_orderb, double>::iterator iob = olb.begin();
			iob != olb.end(); iob++) {

		index<k_ordera> idxa;
		index<k_orderb> idxb(olb.get_index(iob));
		idxb.permute(pinv);

		for(size_t i = 0; i < k_ordera; i++) idxa[i] = idxb[map[i]];
		orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), idxa);
		abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(),
			m_bta.get_bis().get_block_index_dims());
		const transf<k_ordera, double> &tra = oa.get_transf(idxa);

		bool zeroa = ctrla.req_is_zero_block(cidxa.get_index());
		bool zerob = ctrlb.req_is_zero_block(idxb);

		if(zeroa) {
			ctrlb.req_zero_block(idxb);
			continue;
		}

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

		tensor_i<k_ordera, double> &blka =
				ctrla.req_block(cidxa.get_index());
		tensor_i<k_orderb, double> &blkb = ctrlb.req_block(idxb);

		permutation<k_orderb> permb(pb.get_perm());
		permb.permute(m_perm);
		tod_diag<N, M>(blka, m_msk, permb, tra.get_coeff() * c).
				perform(blkb);

		ctrla.ret_block(cidxa.get_index());
		ctrlb.ret_block(idxb);
	}

	btod_diag<N, M>::stop_timer();
}

template<size_t N, size_t M>
void btod_diag<N, M>::do_perform(block_tensor_i<k_orderb, double> &btb,
		const index<k_orderb> &dst_idx, bool zero, double c)
	throw(exception) {

	throw not_implemented(g_ns, k_clazz,
			"do_perform(block_tensor_i<k_orderb, double>&,const index<k_orderb>&,bool,double)",
			__FILE__, __LINE__);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_DIAG_H
