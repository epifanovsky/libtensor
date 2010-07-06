#ifndef LIBTENSOR_TOD_CONTRACT2_IMPL_H
#define LIBTENSOR_TOD_CONTRACT2_IMPL_H

#include "../core/tensor_ctrl.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *tod_contract2<N, M, K>::k_clazz = "tod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
tod_contract2<N, M, K>::tod_contract2(const contraction2<N, M, K> &contr,
	tensor_i<k_ordera, double> &ta, tensor_i<k_orderb, double> &tb) :

	m_contr(contr), m_ta(ta), m_tb(tb) {

}


template<size_t N, size_t M, size_t K>
tod_contract2<N, M, K>::~tod_contract2() {

}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::prefetch() {

	tensor_ctrl<k_ordera, double>(m_ta).req_prefetch();
	tensor_ctrl<k_orderb, double>(m_tb).req_prefetch();
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::perform(tensor_i<k_orderc, double> &tc) {

	do_perform(tc, true, 1.0);
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::perform(tensor_i<k_orderc, double> &tc, double d) {

	do_perform(tc, false, d);
}


template<size_t N, size_t M, size_t K>
void tod_contract2<N, M, K>::do_perform(tensor_i<k_orderc, double> &tc,
	bool zero, double d) {

	typedef typename loop_list_mul::list_t list_t;
	typedef typename loop_list_mul::registers registers_t;
	typedef typename loop_list_mul::node node_t;

	timings< tod_contract2<N, M, K> >::start_timer();

	try {

	tensor_ctrl<k_ordera, double> ca(m_ta);
	tensor_ctrl<k_orderb, double> cb(m_tb);
	tensor_ctrl<k_orderc, double> cc(tc);

	ca.req_prefetch();
	cb.req_prefetch();
	cc.req_prefetch();

	const dimensions<k_ordera> &dimsa = m_ta.get_dims();
	const dimensions<k_orderb> &dimsb = m_tb.get_dims();
	const dimensions<k_orderc> &dimsc = tc.get_dims();

	list_t loop;
	loop_list_adapter list_adapter(loop);
	contraction2_list_builder<N, M, K, loop_list_adapter> lstbld(m_contr);
	lstbld.populate(list_adapter, dimsa, dimsb, dimsc);

	const double *pa = ca.req_const_dataptr();
	const double *pb = cb.req_const_dataptr();
	double *pc = cc.req_dataptr();

	if(zero) {
		timings< tod_contract2<N, M, K> >::start_timer("zero");
		size_t szc = tc.get_dims().get_size();
		for(size_t i = 0; i < szc; i++) pc[i] = 0.0;
		timings< tod_contract2<N, M, K> >::stop_timer("zero");
	}

	registers_t r;
	r.m_ptra[0] = pa;
	r.m_ptra[1] = pb;
	r.m_ptrb[0] = pc;
	r.m_ptra_end[0] = pa + dimsa.get_size();
	r.m_ptra_end[1] = pb + dimsb.get_size();
	r.m_ptrb_end[0] = pc + dimsc.get_size();

//	std::cout << "[";
	loop_list_mul::run_loop(loop, r, d);
//	std::cout << "]" << std::endl;

	ca.ret_const_dataptr(pa);
	cb.ret_const_dataptr(pb);
	cc.ret_dataptr(pc);

	} catch(...) {
		timings< tod_contract2<N, M, K> >::stop_timer();
		throw;
	}

	timings< tod_contract2<N, M, K> >::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_CONTRACT2_IMPL_H
