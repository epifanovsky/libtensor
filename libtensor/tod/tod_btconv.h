#ifndef LIBTENSOR_TOD_BTCONV_H
#define LIBTENSOR_TOD_BTCONV_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../linalg.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/tensor_i.h"
#include "../core/tensor_ctrl.h"
#include "../tod/processor.h"
#include "../btod/transf_double.h"

namespace libtensor {

/**	\brief Unfolds a block %tensor into a simple %tensor
	\tparam N Tensor order.

	\ingroup libtensor_tod
 **/
template<size_t N>
class tod_btconv : public timings<tod_btconv<N> > {
public:
	static const char *k_clazz; //!< Class name

private:
	struct registers {
		const double *m_ptra;
		double *m_ptrb;
	};

	struct loop_list_node;
	typedef std::list<loop_list_node> loop_list_t;
	typedef processor<loop_list_t, registers> processor_t;
	typedef processor_op_i<loop_list_t, registers> processor_op_i_t;

	struct loop_list_node {
	public:
		size_t m_weight;
		size_t m_inca, m_incb;
		processor_op_i_t *m_op;
		loop_list_node()
			: m_weight(0), m_inca(0), m_incb(0), m_op(NULL) { }
		loop_list_node(size_t weight, size_t inca, size_t incb)
			: m_weight(weight), m_inca(inca), m_incb(incb),
			m_op(NULL) { }
		processor_op_i_t *op() const { return m_op; }
	};

	class op_loop : public processor_op_i_t {
	private:
		size_t m_len, m_inca, m_incb;
	public:
		op_loop(size_t len, size_t inca, size_t incb)
			: m_len(len), m_inca(inca), m_incb(incb) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	class op_dcopy : public processor_op_i_t {
	private:
		size_t m_len, m_inca, m_incb;
		double m_c;
	public:
		op_dcopy(size_t len, size_t inca, size_t incb, double c)
			: m_len(len), m_inca(inca), m_incb(incb), m_c(c) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

private:
	block_tensor_i<N, double> &m_bt; //!< Source block %tensor

public:
	//!	\name Construction and destruction
	//@{

	tod_btconv(block_tensor_i<N, double> &bt);
	~tod_btconv();

	//@}

	//!	\name Tensor operation
	//@{

	void perform(tensor_i<N, double> &t) throw(exception);

	//@}

private:
	void copy_block(double *optr, const dimensions<N> &odims,
		const index<N> &ooffs, const double *iptr,
		const dimensions<N> &idims, const permutation<N> &iperm,
		double icoeff) throw(exception);
};


template<size_t N>
const char *tod_btconv<N>::k_clazz = "tod_btconv<N>";


template<size_t N>
tod_btconv<N>::tod_btconv(block_tensor_i<N, double> &bt) : m_bt(bt) {

}

template<size_t N>
tod_btconv<N>::~tod_btconv() {

}

template<size_t N>
void tod_btconv<N>::perform(tensor_i<N, double> &t) throw(exception) {
	tod_btconv<N>::start_timer();

	static const char *method = "perform(tensor_i<N, double>&)";

	const block_index_space<N> &bis = m_bt.get_bis();
	dimensions<N> bidims(bis.get_block_index_dims());
	if(!bis.get_dims().equals(t.get_dims())) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Incorrect dimensions of the output tensor.");
	}

	block_tensor_ctrl<N, double> src_ctrl(m_bt);
	tensor_ctrl<N, double> dst_ctrl(t);

	double *dst_ptr = dst_ctrl.req_dataptr();
	size_t sz = t.get_dims().get_size();
	for(register size_t i = 0; i < sz; i++) dst_ptr[i] = 0.0;

	orbit_list<N, double> orblst(src_ctrl.req_symmetry());
	typename orbit_list<N, double>::iterator iorbit = orblst.begin();
	for(; iorbit != orblst.end(); iorbit++) {

		orbit<N, double> orb(src_ctrl.req_symmetry(),
			orblst.get_index(iorbit));
		index<N> blk_idx;
		bidims.abs_index(orb.get_abs_canonical_index(), blk_idx);
		if(src_ctrl.req_is_zero_block(blk_idx)) continue;

		tensor_i<N, double> &blk = src_ctrl.req_block(blk_idx);
		tensor_ctrl<N, double> blk_ctrl(blk);
		const double *src_ptr = blk_ctrl.req_const_dataptr();

		typename orbit<N, double>::iterator i = orb.begin();
		while(i != orb.end()) {
			index<N> idx;
			bidims.abs_index(i->first, idx);
			const transf<N, double> &tr = i->second;
			index<N> dst_offset = bis.get_block_start(idx);
			copy_block(dst_ptr, t.get_dims(), dst_offset,
				src_ptr, blk.get_dims(),
				tr.get_perm(), tr.get_coeff());
			i++;
		}

		blk_ctrl.ret_dataptr(src_ptr);
		src_ctrl.ret_block(blk_idx);

	}

	dst_ctrl.ret_dataptr(dst_ptr);

	tod_btconv<N>::stop_timer();
}

template<size_t N>
void tod_btconv<N>::copy_block(double *optr, const dimensions<N> &odims,
	const index<N> &ooffs, const double *iptr, const dimensions<N> &idims,
	const permutation<N> &iperm, double icoeff) throw(exception) {

	permutation<N> inv_perm(iperm);
	inv_perm.invert();
	size_t ib[N];
	for(size_t i = 0; i < N; i++) ib[i] = i;
	inv_perm.apply(ib);

	loop_list_t lst;
	for(size_t i = 0; i < N; i++) {
		size_t inca = idims.get_increment(i);
		size_t incb = odims.get_increment(ib[i]);
		loop_list_node node(idims[i], inca, incb);
		if(i == N - 1) {
			node.m_op = new op_dcopy(idims[i], inca, incb, icoeff);
		} else {
			node.m_op = new op_loop(idims[i], inca, incb);
		}
		lst.push_back(node);
	}

	registers regs;
	regs.m_ptra = iptr;
	regs.m_ptrb = optr + odims.abs_index(ooffs);
	processor_t proc(lst, regs);
	proc.process_next();

	for(typename loop_list_t::iterator i = lst.begin();
		i != lst.end(); i++) {

		delete i->m_op;
		i->m_op = NULL;
	}
}

template<size_t N>
void tod_btconv<N>::op_loop::exec(processor_t &proc, registers &regs)
	throw(exception) {

	const double *ptra = regs.m_ptra;
	double *ptrb = regs.m_ptrb;

	for(size_t i=0; i<m_len; i++) {
		regs.m_ptra = ptra;
		regs.m_ptrb = ptrb;
		proc.process_next();
		ptra += m_inca;
		ptrb += m_incb;
	}
}

template<size_t N>
void tod_btconv<N>::op_dcopy::exec(processor_t &proc, registers &regs)
	throw(exception) {
	blas_dcopy(m_len, regs.m_ptra, m_inca, regs.m_ptrb, m_incb);
	if(m_c != 1.0) {
		blas_dscal(m_len, m_c, regs.m_ptrb, m_incb);
	}
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_BTCONV_H
