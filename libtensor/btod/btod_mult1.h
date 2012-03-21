#ifndef LIBTENSOR_BTOD_MULT1_H
#define LIBTENSOR_BTOD_MULT1_H

#include "../defs.h"
#include "../core/abs_index.h"
#include "../core/block_index_space_product_builder.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../core/permutation_builder.h"
#include "../symmetry/so_copy.h"
#include "../symmetry/so_dirprod.h"
#include "../symmetry/so_merge.h"
#include "../tod/tod_copy.h"
#include "../tod/tod_mult1.h"
#include "../tod/tod_set.h"
#include "bad_block_index_space.h"

namespace libtensor {


/**	\brief Elementwise multiplication of two block tensors
	\tparam N Tensor order.

	\sa tod_mult1

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_mult1 :
	public timings< btod_mult1<N> >{
public:
	static const char *k_clazz; //!< Class name

private:
	block_tensor_i<N, double> &m_btb; //!< Second argument
	permutation<N> m_pb; //!< Permutation of second argument
	bool m_recip; //!< Reciprocal
	double m_c; //!< Scaling coefficient

public:
	btod_mult1(block_tensor_i<N, double> &btb,
			bool recip = false, double c = 1.0);

	btod_mult1(block_tensor_i<N, double> &btb, const permutation<N> &pb,
			bool recip = false, double c = 1.0);

	void perform(block_tensor_i<N, double> &btc);

	void perform(block_tensor_i<N, double> &btc, double c);

protected:
	void compute_block(dense_tensor_i<N, double> &blk, const index<N> &idx);


private:
	void do_perform(block_tensor_i<N, double> &btc, bool zero, double c);

private:
	btod_mult1(const btod_mult1<N> &);
	const btod_mult1<N> &operator=(const btod_mult1<N> &);

};


template<size_t N>
const char *btod_mult1<N>::k_clazz = "btod_mult1<N>";


template<size_t N>
btod_mult1<N>::btod_mult1(
		block_tensor_i<N, double> &btb, bool recip, double c) :

	m_btb(btb), m_recip(recip), m_c(c) {
}

template<size_t N>
btod_mult1<N>::btod_mult1(
		block_tensor_i<N, double> &btb, const permutation<N> &pb,
		bool recip, double c) :

	m_btb(btb), m_pb(pb), m_recip(recip), m_c(c) {
}

template<size_t N>
void btod_mult1<N>::perform(block_tensor_i<N, double> &bta) {

	static const char *method = "perform(block_tensor_i<N, double>&)";

	if(!bta.get_bis().equals(m_btb.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "bta");
	}

	do_perform(bta, true, 1.0);
}


template<size_t N>
void btod_mult1<N>::perform(block_tensor_i<N, double> &bta, double c) {

	static const char *method =
		"perform(block_tensor_i<N, double>&, double)";

	if(!bta.get_bis().equals(m_btb.get_bis())) {
		throw bad_block_index_space(g_ns, k_clazz, method,
			__FILE__, __LINE__, "bta");
	}

	do_perform(bta, false, c);
}


template<size_t N>
void btod_mult1<N>::do_perform(
	block_tensor_i<N, double> &bta, bool zero, double c) {

	static const char *method =
		"do_perform(block_tensor_i<N, double>&, bool, double)";

	cpu_pool cpus(1);

	btod_mult1::start_timer();

	block_tensor_ctrl<N, double> ctrla(bta), ctrlb(m_btb);

	// Copy sym(A) and permuted sym(B) and install \sym(A) \cap \sym(B) in A

    symmetry<N, double> syma(bta.get_bis());
    so_copy<N, double>(ctrla.req_const_symmetry()).perform(syma);

    sequence<N, size_t> seq2a;
    for (size_t i = 0; i < N; i++) {
        seq2a[i] = i + N;
    }
    m_pb.apply(seq2a);

    sequence<N + N, size_t> seq1b, seq2b;
    for (size_t i = 0; i < N; i++) {
        seq1b[i] = seq2b[i] = i;
    }
    for (size_t i = N; i < N + N; i++) {
        seq1b[i] = i; seq2b[i] = seq2a[i - N];
    }
    permutation_builder<N + N> pbb(seq2b, seq1b);

    block_index_space_product_builder<N, N> bbx(bta.get_bis(), bta.get_bis(),
            permutation<N + N>());

    symmetry<N + N, double> symx(bbx.get_bis());
    so_dirprod<N, N, double>(syma,
            ctrlb.req_const_symmetry(), pbb.get_perm()).perform(symx);
    so_merge<N + N, N + N, N, double> merge(symx);
    for (size_t i = 0; i < N; i++) {
        mask<N + N> m;
        m[i] = m[i + N] = true;
        merge.add_mask(m);
    }
    merge.perform(ctrla.req_symmetry());

	// First loop over all orbits in sym(A) \cap sym(B) and copy blocks which
	// were not canonical in sym(A)

	orbit_list<N, double> ol(ctrla.req_symmetry());

	for(typename orbit_list<N, double>::iterator io = ol.begin();
		io != ol.end(); io++) {

		index<N> idx(ol.get_index(io));

		orbit<N, double> oa(syma, idx);
		abs_index<N> cidxa(oa.get_abs_canonical_index(),
				bta.get_bis().get_block_index_dims());

		if (idx.equals(cidxa.get_index()))
			continue;

		if (ctrla.req_is_zero_block(cidxa.get_index()))
			continue;

		dense_tensor_i<N, double> &blk = ctrla.req_block(idx);
		dense_tensor_i<N, double> &blka = ctrla.req_block(cidxa.get_index());

		const transf<N, double> &tra = oa.get_transf(idx);

		tod_copy<N>(blka, tra.get_perm(), tra.get_coeff()).perform(cpus, true, 1.0, blk);

		ctrla.ret_block(cidxa.get_index());
		ctrla.ret_block(idx);
	}

	// Second loop over all orbits in sym(A) \cap sym(B) and do the operation

	permutation<N> pinvb(m_pb, true);

	for(typename orbit_list<N, double>::iterator ioa = ol.begin();
		ioa != ol.end(); ioa++) {

		index<N> idxa(ol.get_index(ioa)), idxb(idxa);

		idxb.permute(pinvb);
		orbit<N, double> ob(ctrlb.req_const_symmetry(), idxb);
		abs_index<N> cidxb(ob.get_abs_canonical_index(),
				m_btb.get_bis().get_block_index_dims());

		bool zeroa = ctrla.req_is_zero_block(idxa);
		bool zerob = ctrlb.req_is_zero_block(cidxb.get_index());

		if(m_recip && zerob) {
			throw bad_parameter(g_ns, k_clazz, method,
				__FILE__, __LINE__, "zero in btb");
		}

		if(zero && (zeroa || zerob)) {
			ctrla.req_zero_block(idxa);
			continue;
		}

		if (zeroa || zerob)
			continue;

		dense_tensor_i<N, double> &blka = ctrla.req_block(idxa);
		dense_tensor_i<N, double> &blkb = ctrlb.req_block(cidxb.get_index());

		const transf<N, double> &trb = ob.get_transf(idxb);
		double k = m_c;
		if (m_recip) k /= trb.get_coeff();
		else k *= trb.get_coeff();

		permutation<N> pb(trb.get_perm());
		pb.permute(m_pb);

		if(zero) {
			tod_mult1<N>(blkb, pb, m_recip, k * c).perform(blka);
		} else {
			tod_mult1<N>(blkb, pb, m_recip, k).perform(blka, c);
		}

		ctrla.ret_block(idxa);
		ctrlb.ret_block(cidxb.get_index());
	}

	btod_mult1::stop_timer();

}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT1_H
