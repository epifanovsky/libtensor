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
#include "../tod/tod_diag.h"
#include "../tod/tod_set.h"
#include "bad_block_index_space.h"
#include "btod_additive.h"
#include "transf_double.h"

namespace libtensor {


/**	\brief Extracts a general diagonal from a block %tensor
	\tparam N Tensor order.
	\tparam M Diagonal order.

	\ingroup libtensor_btod
 **/
template<size_t N, size_t M>
class btod_diag :
	public btod_additive<N - M + 1>,
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

public:
	btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
		double c = 1.0);

	btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
		const permutation<N - M + 1> &p, double c = 1.0);

	virtual const block_index_space<k_orderb> &get_bis() const {
		return m_bis;
	}

	virtual const symmetry<k_orderb, double> &get_symmetry() const {
		throw not_implemented(g_ns, k_clazz, "get_symmetry()",
			__FILE__, __LINE__);
	}

	virtual void perform(block_tensor_i<k_orderb, double> &btb)
		throw(exception);

	virtual void perform(block_tensor_i<k_orderb, double> &btb, double c)
		throw(exception);

	virtual void perform(block_tensor_i<k_orderb, double> &btb,
		const index<k_orderb> &idx) throw(exception) {

		throw not_implemented(g_ns, k_clazz, "perform()",
			__FILE__, __LINE__);
	}

private:
	/**	\brief Forms the block %index space of the output or throws an
			exception if the input is incorrect
	 **/
	static block_index_space<N - M + 1> mk_bis(
		const block_index_space<N> &bis, const mask<N> &msk);

	void do_perform(block_tensor_i<k_orderb, double> &btb, bool zero,
		double c);

private:
	btod_diag(const btod_diag<N, M>&);
	const btod_diag<N, M> &operator=(const btod_diag<N, M>&);

};


template<size_t N, size_t M>
const char *btod_diag<N, M>::k_clazz = "btod_diag<N, M>";


template<size_t N, size_t M>
btod_diag<N, M>::btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
	double c) :

	m_bta(bta), m_msk(m), m_c(c), m_bis(mk_bis(bta.get_bis(), m_msk)) {

}


template<size_t N, size_t M>
btod_diag<N, M>::btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
	const permutation<N - M + 1> &p, double c) :

	m_bta(bta), m_msk(m), m_perm(p), m_c(c),
	m_bis(mk_bis(bta.get_bis(), m_msk)) {

	m_bis.permute(p);
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

	do_perform(btb, true, 1.0);
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

	do_perform(btb, false, c);
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

	return obis;
}


template<size_t N, size_t M>
void btod_diag<N, M>::do_perform(
	block_tensor_i<k_orderb, double> &btb, bool zero, double c) {

	btod_diag<N, M>::start_timer();

	block_tensor_ctrl<k_ordera, double> ctrla(m_bta);
	block_tensor_ctrl<k_orderb, double> ctrlb(btb);

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

	orbit_list<k_orderb, double> olstb(ctrlb.req_symmetry());

	for(typename orbit_list<k_orderb, double>::iterator iob = olstb.begin();
		iob != olstb.end(); iob++) {

		index<k_ordera> idxa;
		index<k_orderb> idxb(olstb.get_index(iob));
		idxb.permute(pinv);

		for(size_t i = 0; i < k_ordera; i++) idxa[i] = idxb[map[i]];
		orbit<k_ordera, double> oa(ctrla.req_symmetry(), idxa);
		abs_index<k_ordera> idxa1(oa.get_abs_canonical_index(),
			m_bta.get_bis().get_block_index_dims());
		const transf<k_ordera, double> &tra = oa.get_transf(idxa);

		bool zeroa = ctrla.req_is_zero_block(idxa1.get_index());

		if(zero && zeroa) {
			ctrlb.req_zero_block(idxb);
			continue;
		}
		if(zeroa) continue;

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

		tensor_i<k_ordera, double> &blka = ctrla.req_block(
			idxa1.get_index());
		tensor_i<k_orderb, double> &blkb = ctrlb.req_block(idxb);

		permutation<k_orderb> permb(pb.get_perm());
		permb.permute(m_perm);
		if(zero) {
			tod_diag<N, M>(blka, m_msk, permb, tra.get_coeff() * c).
				perform(blkb);
		} else {
			tod_diag<N, M>(blka, m_msk, permb, tra.get_coeff()).
				perform(blkb, c);
		}

		ctrla.ret_block(idxa1.get_index());
		ctrlb.ret_block(idxb);
	}

	btod_diag<N, M>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_DIAG_H
