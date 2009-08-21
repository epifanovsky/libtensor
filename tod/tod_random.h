#ifndef LIBTENSOR_TOD_RANDOM_H
#define LIBTENSOR_TOD_RANDOM_H

#include <time.h>
#include "defs.h"
#include "exception.h"
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
	virtual void perform(tensor_i<N, double> &t) throw(exception);
	//@}

	//!	\name Implementation of libtensor::tod_additive<N>
	//@{
	virtual void perform(tensor_i<N, double> &t, double c) throw(exception);
	//@}
private:
	void do_perform( tensor_i<N,double>& t, double c ) throw(exception);
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
void tod_random<N>::perform(tensor_i<N, double> &t) throw(exception) {
	tensor_ctrl<N,double> ctrl(t);
	double* ptr=ctrl.req_dataptr();
	size_t total_size=t.get_dims().get_size();
	
	for (size_t i=0; i<total_size; i++) ptr[i]=drand48();
		
	ctrl.ret_dataptr(ptr); 
}

template<size_t N> 
void tod_random<N>::perform(tensor_i<N, double> &t, double c) throw(exception) {
	tensor_ctrl<N,double> ctrl(t);
	double* ptr=ctrl.req_dataptr();
	size_t total_size=t.get_dims().get_size();
	
	for (size_t i=0; i<total_size; i++) ptr[i]+=c*drand48();
		
	ctrl.ret_dataptr(ptr); 
}

} // namespace libtensor

#endif // LIBTENSOR_TOD_RANDOM_H_
