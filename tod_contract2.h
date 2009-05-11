#ifndef LIBTENSOR_TOD_CONTRACT2_H
#define LIBTENSOR_TOD_CONTRACT2_H

#include "defs.h"
#include "exception.h"
#include "tod_additive.h"
#include "contraction2.h"
#include "contraction2_list.h"
#include "contraction2_processor.h"

namespace libtensor {

/**	\brief Contracts two tensors (double)

	\tparam N Order of the first %tensor (a) less the contraction degree
	\tparam M Order of the second %tensor (b) less the contraction degree
	\tparam K Contraction degree (the number of indexes over which the
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
	contraction2<N,M,K> m_contr; //!< Contraction
	tensor_i<N+K,double> &m_ta; //!< First tensor (a)
	tensor_i<M+K,double> &m_tb; //!< Second tensor (b)

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the contraction operation

		\param contr Contraction.
		\param ta Tensor a (first argument).
		\param tb Tensor b (second argument).
	 **/
	tod_contract2(const contraction2<N,M,K> &contr,
		tensor_i<N+K,double> &ta, tensor_i<M+K,double> &tb);


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
//	bool check_dims_ab();

	/**	\brief Check if the resulting tensor (c) has compatible
			dimensions
	**/
//	bool check_dims_c(const dimensions<N+M> &dc);
};

template<size_t N, size_t M, size_t K>
tod_contract2<N,M,K>::tod_contract2(const contraction2<N,M,K> &contr,
	tensor_i<N+K,double> &ta, tensor_i<M+K,double> &tb) :
	m_contr(contr), m_ta(ta), m_tb(tb) {
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
void tod_contract2<N,M,K>::perform(tensor_i<N+M,double> &tc) throw(exception) {
	contraction2_list<N+M+K> list;
	m_contr.populate(list, m_ta.get_dims(), m_tb.get_dims(), tc.get_dims());

	tensor_ctrl<N+K,double> ctrla(m_ta);
	tensor_ctrl<M+K,double> ctrlb(m_tb);
	tensor_ctrl<N+M,double> ctrlc(tc);

	const double *ptra = ctrla.req_const_dataptr();
	const double *ptrb = ctrlb.req_const_dataptr();
	double *ptrc = ctrlc.req_dataptr();

	contraction2_processor<N+M+K>(list, ptrc, ptra, ptrb).contract();

	ctrla.ret_dataptr(ptra);
	ctrlb.ret_dataptr(ptrb);
	ctrlc.ret_dataptr(ptrc);
}

template<size_t N, size_t M, size_t K>
void tod_contract2<N,M,K>::perform(tensor_i<N+M,double> &t, double d)
	throw(exception) {

//	if(!check_dims_c(t.get_dims())) {
//		throw_exc("tod_contract2<N,M,K>",
//			"perform(tensor_i<N+M,double>&, double)",
//			"Incompatible dimensions of tensor c");
//	}

	tensor_ctrl<N+K,double> ctrla(m_ta);
	tensor_ctrl<M+K,double> ctrlb(m_tb);
	tensor_ctrl<N+M,double> ctrlc(t);

	const double *ptra = ctrla.req_const_dataptr();
	const double *ptrb = ctrlb.req_const_dataptr();
	double *ptrc = ctrlc.req_dataptr();


	ctrla.ret_dataptr(ptra);
	ctrlb.ret_dataptr(ptrb);
	ctrlc.ret_dataptr(ptrc);
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_H

