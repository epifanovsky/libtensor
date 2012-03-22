#ifndef LIBTENSOR_LOOP_LIST_RUNNER_H
#define LIBTENSOR_LOOP_LIST_RUNNER_H

#include "kernel_base.h"

namespace libtensor {


/**
	\ingroup libtensor_tod_kernel
 **/
template<size_t N, size_t M>
class loop_list_runner {
public:
	typedef std::list< loop_list_node<N, M> > list_t;
	typedef typename list_t::const_iterator iterator_t;

private:
	const list_t &m_list;

public:
	loop_list_runner(const list_t &list) : m_list(list) { }

	void run(const loop_registers<N, M> &r, kernel_base<N, M> &k);

private:
	void run_loop(const iterator_t &i, const loop_registers<N, M> &r,
		kernel_base<N, M> &k);

};


template<size_t N, size_t M>
inline void loop_list_runner<N, M>::run(const loop_registers<N, M> &r,
	kernel_base<N, M> &k) {

	run_loop(m_list.begin(), r, k);
}


template<size_t N, size_t M>
void loop_list_runner<N, M>::run_loop(const iterator_t &i,
	const loop_registers<N, M> &r, kernel_base<N, M> &k) {

	if(i == m_list.end()) {
		k.run(r);
		return;
	}

	loop_registers<N, M> r1(r);
	for(size_t j = 0; j < i->weight(); j++) {
		iterator_t ii = i; ii++;
		run_loop(ii, r1, k);
		for(size_t k = 0; k < N; k++) {
			r1.m_ptra[k] += i->stepa(k);
		}
		for(size_t k = 0; k < M; k++) {
			r1.m_ptrb[k] += i->stepb(k);
		}
	}
}


} // namespace libtensor

#endif // LIBTENSOR_LOOP_LIST_RUNNER_H
