#ifndef LIBTENSOR_BTOD_TRACE_H
#define LIBTENSOR_BTOD_TRACE_H

#include "../defs.h"
#include "../not_implemented.h"
#include "../core/abs_index.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../core/permutation_builder.h"
#include "../tod/tod_trace.h"
#include "bad_block_index_space.h"
#include "transf_double.h"

namespace libtensor {


/**	\brief Computes the trace of a matricized block %tensor
	\tparam N Tensor diagonal order.

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_trace : public timings< btod_trace<N> > {
public:
	static const char *k_clazz; //!< Class name

public:
	static const size_t k_ordera = 2 * N; //!< Order of the argument

private:
	block_tensor_i<k_ordera, double> &m_bta; //!< Input block %tensor
	permutation<k_ordera> m_perm; //!< Permutation of the %tensor

public:
	btod_trace(block_tensor_i<k_ordera, double> &bta);

	btod_trace(block_tensor_i<k_ordera, double> &bta,
		const permutation<k_ordera> &perm);

	double calculate();

private:
	btod_trace(const btod_trace<N>&);
	const btod_trace<N> &operator=(const btod_trace<N>&);

};


template<size_t N>
const char *btod_trace<N>::k_clazz = "btod_trace<N>";


template<size_t N>
btod_trace<N>::btod_trace(block_tensor_i<k_ordera, double> &bta) : m_bta(bta) {

}


template<size_t N>
btod_trace<N>::btod_trace(block_tensor_i<k_ordera, double> &bta,
	const permutation<k_ordera> &perm) : m_bta(bta), m_perm(perm) {

}


template<size_t N>
double btod_trace<N>::calculate() {

	static const char *method = "calculate()";

	double tr = 0;

	btod_trace<N>::start_timer();

	try {

	dimensions<k_ordera> bidimsa = m_bta.get_bis().get_block_index_dims();

	block_tensor_ctrl<k_ordera, double> ca(m_bta);

	orbit_list<k_ordera, double> ola(ca.req_const_symmetry());
	for(typename orbit_list<k_ordera, double>::iterator ioa = ola.begin();
		ioa != ola.end(); ioa++) {

	if(ca.req_is_zero_block(ola.get_index(ioa))) continue;

	tensor_i<k_ordera, double> *ba = 0;

	orbit<k_ordera, double> oa(ca.req_const_symmetry(), ola.get_index(ioa));
	for(typename orbit<k_ordera, double>::iterator iia = oa.begin();
		iia != oa.end(); iia++) {

		abs_index<k_ordera> aia(oa.get_abs_index(iia), bidimsa);
		index<k_ordera> ia(aia.get_index()); ia.permute(m_perm);

		bool skip = false;
		for(register size_t i = 0; i < N; i++) if(ia[i] != ia[N + i]) {
			skip = true;
			break;
		}
		if(skip) continue;

		transf<k_ordera, double> tra(oa.get_transf(iia));
		tra.permute(m_perm);

		if(ba == 0) ba = &ca.req_block(ola.get_index(ioa));
		double tr0 = tod_trace<N>(*ba, tra.get_perm()).calculate();
		tr += tr0 * tra.get_coeff();
	}

	if(ba != 0) ca.ret_block(ola.get_index(ioa));

	}

	} catch(...) {
		btod_trace<N>::stop_timer();
		throw;
	}

	btod_trace<N>::stop_timer();

	return tr;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_TRACE_H
