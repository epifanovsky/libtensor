#ifndef LIBTENSOR_TOD_RANDOM_H
#define LIBTENSOR_TOD_RANDOM_H

#include <ctime>
#include <cstdlib>
#include "../defs.h"
#include "../exception.h"
#include "../core/tensor_ctrl.h"
#include "../mp/auto_cpu_lock.h"
#include "tod_additive.h"

namespace libtensor {

/** \brief Fills a %tensor with random numbers or adds them to it

	\tparam N Tensor order

	This operation either fills a %tensor with random numbers equally
	distributed in the intervall [0;1[ or adds those numbers to the %tensor
	scaled bei a coefficient

 **/
template<size_t N>
class tod_random : public tod_additive<N> {
private:
	static void update_seed(); //! updates the seed value by using srand48
public:
	//! Constructurs and destructors
	//@{

	//! \brief Prepares the operation
	tod_random();

	//! \brief Virtual destructor
	virtual ~tod_random();

	//@}

	//!	\name Implementation of
	//!		libtensor::direct_tensor_operation<N, double>
	//@{
	virtual void prefetch() throw(exception);
    virtual void perform(cpu_pool &cpus, bool zero, double c,
        dense_tensor_i<N, double> &t);
	void perform(cpu_pool &cpus, dense_tensor_i<N, double> &t);
    void perform(cpu_pool &cpus, dense_tensor_i<N, double> &t, double c);
	//@}

private:
	void do_perform( dense_tensor_i<N,double>& t, double c ) throw(exception);
};

template<size_t N>
inline tod_random<N>::tod_random() {
	update_seed();
}

template<size_t N>
tod_random<N>::~tod_random() {
}

template<size_t N>
void tod_random<N>::update_seed() {
	static time_t timestamp=time(NULL);
	static long seed=timestamp;
	if ( time(NULL)-timestamp > 60 ) {
		timestamp=time(NULL);
		seed+=timestamp+lrand48();
		srand48(seed);
	}
}

template<size_t N>
void tod_random<N>::prefetch() throw(exception) {
}

template<size_t N>
void tod_random<N>::perform(cpu_pool &cpus, dense_tensor_i<N, double> &t) {

    perform(cpus, true, 1.0, t);
}

template<size_t N>
void tod_random<N>::perform(cpu_pool &cpus, dense_tensor_i<N, double> &t, double c) {

    perform(cpus, false, c, t);
}

template<size_t N>
void tod_random<N>::perform(cpu_pool &cpus, bool zero, double c,
    dense_tensor_i<N, double> &t) {

    tensor_ctrl<N, double> ctrl(t);
    size_t sz = t.get_dims().get_size();
    double *ptr = ctrl.req_dataptr();

    {
        auto_cpu_lock cpu(cpus);

        if(zero) {
            for(size_t i = 0; i < sz; i++) ptr[i] = c * drand48();
        } else {
            for(size_t i = 0; i < sz; i++) ptr[i] += c * drand48();
        }
    }

    ctrl.ret_dataptr(ptr);
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_RANDOM_H_
