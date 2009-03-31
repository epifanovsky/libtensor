#ifndef LIBTENSOR_TOD_CONTRACT2_H
#define LIBTENSOR_TOD_CONTRACT2_H

#include "defs.h"
#include "exception.h"
#include "tod_additive.h"
#include "tod_contract2_impl.h"

namespace libtensor {

/**	\brief Contracts two tensors (double)

	\param N Order of the first %tensor (a) less the contraction degree
	\param M Order of the second %tensor (b) less the contraction degree
	\param K Contraction degree (the number of indexes over which the
		tensors are contracted)

	This operation contracts %tensor T1 permuted as P1 with %tensor T2
	permuted as P2 over n last indexes. The result is permuted as Pres
	and written or added to the resulting %tensor.

	Although it is convenient to define a contraction through permutations,
	it is not the most efficient way of calculating it. This class seeks
	to use algorithms tailored for different tensors to get the best
	performance. For more information, read the wiki section on %tensor
	contractions.

	\ingroup libtensor_tod
**/
template<size_t N, size_t M, size_t K>
class tod_contract2 : public tod_additive<N+M> {
private:
	tensor_i<N+K,double> &m_ta; //!< First tensor (a)
	tensor_i<M+K,double> &m_tb; //!< Second tensor (b)
	permutation<N+K> m_pa; //!< Permutation of the first %tensor (a)
	permutation<M+K> m_pb; //!< Permutation of the second %tensor (b)
	permutation<N+M> m_pc; //!< Permutation of the result (c)

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the operation

		\param ta Tensor a
		\param pa Permutation of %tensor a
		\param tb Tensor b
		\param pb Permutation of %tensor b
		\param pc Permutation of the resulting %tensor c
	**/
	tod_contract2(tensor_i<N+K,double> &ta, const permutation<N+K> &pa,
		tensor_i<M+K,double> &tb, const permutation<M+K> &pb,
		const permutation<N+M> &pc) throw(exception);

	/**	\brief Virtual destructor
	**/
	virtual ~tod_contract2();

	//@}

	//!	\name Implementation of direct_tensor_operation<T>
	//@{
	virtual void prefetch() throw(exception);
	//@}

	//!	\name Implementation of tod_additive
	//@{
	virtual void perform(tensor_i<N+M,double> &t) throw(exception);
	virtual void perform(tensor_i<N+M,double> &t, double d)
		throw(exception);
	//@}

private:
	/**	\brief Check if the two tensors to be contracted (a and b) have
			compatible dimensions
	**/
	bool check_dims_ab();

	/**	\brief Check if the resulting tensor (c) has compatible
			dimensions
	**/
	bool check_dims_c(const dimensions<N+M> &dc);
};

template<size_t N, size_t M, size_t K>
tod_contract2<N,M,K>::tod_contract2(
	tensor_i<N+K,double> &ta, const permutation<N+K> &pa,
	tensor_i<M+K,double> &tb, const permutation<M+K> &pb,
	const permutation<N+M> &pc) throw(exception) :
		m_ta(ta), m_pa(pa), m_tb(tb), m_pb(pb), m_pc(pc) {

	if(!check_dims_ab()) {
		throw_exc("tod_contract2<N,M,K>", "tod_contract2()",
			"Incompatible dimensions of tensors a and b");
	}
}

template<size_t N, size_t M, size_t K>
tod_contract2<N,M,K>::~tod_contract2() {
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N,M,K>::prefetch() throw(exception) {
	tensor_ctrl<N+K,double> ctrl_ta(m_ta);
	tensor_ctrl<M+K,double> ctrl_tb(m_tb);
	ctrl_ta.req_prefetch();
	ctrl_tb.req_prefetch();
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N,M,K>::perform(tensor_i<N+M,double> &t) throw(exception) {
	if(!check_dims_c(t.get_dims())) {
		throw_exc("tod_contract2<N,M,K>",
			"perform(tensor_i<N+M,double>&)",
			"Incompatible dimensions of tensor c");
	}

	tensor_ctrl<N+K,double> ctrla(m_ta);
	tensor_ctrl<M+K,double> ctrlb(m_tb);
	tensor_ctrl<N+M,double> ctrlc(t);

	const double *ptra = ctrla.req_const_dataptr();
	const double *ptrb = ctrlb.req_const_dataptr();
	double *ptrc = ctrlc.req_dataptr();

	if(N<=M) {
		tod_contract2_impl<N,M,K>::contract(ptrc, t.get_dims(), m_pc,
			ptra, m_ta.get_dims(), m_pa, ptrb, m_tb.get_dims(),
			m_pb);
	} else {
		tod_contract2_impl<M,N,K>::contract(ptrc, t.get_dims(), m_pc,
			ptrb, m_tb.get_dims(), m_pb, ptra, m_ta.get_dims(),
			m_pa);
	}

	ctrla.ret_dataptr(da);
	ctrlb.ret_dataptr(db);
	ctrlc.ret_dataptr(dc);
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N,M,K>::perform(tensor_i<N+M,double> &t, double d)
	throw(exception) {

	if(!check_dims_c(t.get_dims())) {
		throw_exc("tod_contract2<N,M,K>",
			"perform(tensor_i<N+M,double>&, double)",
			"Incompatible dimensions of tensor c");
	}

	tensor_ctrl<N+K,double> ctrla(m_ta);
	tensor_ctrl<M+K,double> ctrlb(m_tb);
	tensor_ctrl<N+M,double> ctrlc(t);

	const double *ptra = ctrla.req_const_dataptr();
	const double *ptrb = ctrlb.req_const_dataptr();
	double *ptrc = ctrlc.req_dataptr();

	if(N<=M) {
		tod_contract2_impl<N,M,K>::contract(ptrc, t.get_dims(), m_pc,
			ptra, m_ta.get_dims(), m_pa, ptrb, m_tb.get_dims(),
			m_pb, d);
	} else {
		tod_contract2_impl<M,N,K>::contract(ptrc, t.get_dims(), m_pc,
			ptrb, m_tb.get_dims(), m_pb, ptra, m_ta.get_dims(),
			m_pa, d);
	}

	ctrla.ret_dataptr(da);
	ctrlb.ret_dataptr(db);
	ctrlc.ret_dataptr(dc);
}

template<size_t N, size_t M, size_t K>
bool tod_contract2<N,M,K>::check_dims_ab() {
	dimensions<N+K> da(m_ta.get_dims()); da.permute(m_pa);
	dimensions<M+K> db(m_tb.get_dims()); db.permute(m_pb);
	for(size_t i=0; i<K; i++) if(da[N+i]!=db[M+i]) return false;
	return true;
}

template<size_t N, size_t M, size_t K>
bool tod_contract2<N,M,K>::check_dims_c(const dimensions<N+M> &dc) {
	dimensions<N+K> da(m_ta.get_dims()); da.permute(m_pa);
	dimensions<M+K> db(m_tb.get_dims()); db.permute(m_pb);
	dimensions<N+M> dcc(dc); dcc.permute(permutation<N+M>(m_pc).invert());
	register size_t i = 0;
	for(register size_t j=0; j<N; j++,i++) if(da[j]!=dcc[i]) return false;
	for(register size_t j=0; j<M; j++,i++) if(db[j]!=dcc[i]) return false;
	return true;
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_H

