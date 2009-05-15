#ifndef LIBTENSOR_BTOD_CONTRACT2_H
#define LIBTENSOR_BTOD_CONTRACT2_H

#include "defs.h"
#include "exception.h"
#include "block_tensor_i.h"
#include "block_tensor_ctrl.h"
#include "btod_additive.h"
#include "contraction2.h"
#include "tod_contract2.h"

namespace libtensor {

/**	\brief Operation for the contraction of two block tensors

	\ingroup libtensor
 **/
template<size_t N, size_t M, size_t K>
class btod_contract2 : public btod_additive<N + M> {
private:
	static const size_t k_ordera = N + K;
	static const size_t k_orderb = M + K;
	static const size_t k_orderc = N + M;

private:
	contraction2<N, M, K> m_contr; //!< Contraction
	block_tensor_i<k_ordera, double> &m_bta; //!< First argument (a)
	block_tensor_i<k_orderb, double> &m_btb; //!< Second argument (b)

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
	virtual void perform(block_tensor_i<k_orderc, double> &btc)
		throw(exception);
	//@}
};

template<size_t N, size_t M, size_t K>
btod_contract2<N, M, K>::btod_contract2(const contraction2<N, M, K> &contr,
	block_tensor_i<k_ordera, double> &bta,
	block_tensor_i<k_orderb, double> &btb) :
		m_contr(contr), m_bta(bta), m_btb(btb) {
}

template<size_t N, size_t M, size_t K>
btod_contract2<N, M, K>::~btod_contract2() {
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

} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_H
