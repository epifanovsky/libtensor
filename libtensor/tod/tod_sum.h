#ifndef LIBTENSOR_TOD_SUM_H
#define LIBTENSOR_TOD_SUM_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "tod_additive.h"
#include <libtensor/dense_tensor/tod_set.h>

namespace libtensor {

/**	\brief Accumulates the result of a sequence of operations (double)
	\tparam N Tensor order.

	Invokes a series of additive %tensor operations to produce the sum of
	their results in the output %tensor.

	The sequence must contain at least one operation that is passed through
	the constructor. Subsequent operations are passed using add_op().

	Calling perform() will run the operations. Both additive and replacing
	interfaces are available.

	\ingroup libtensor_tod
 **/
template<size_t N>
class tod_sum : public tod_additive<N> {
private:
	struct node {
		tod_additive<N> &m_op;
		double m_c;

		node(tod_additive<N> &op, double c) : m_op(op), m_c(c) { }
	};

private:
	std::list<node> m_lst; //!< List of operations

public:
	//!	\name Construction, destruction, initialization
	//@{

	/**	\brief Initializes the operation
	 **/
	tod_sum(tod_additive<N> &op, double c = 1.0);

	/**	\brief Virtual destructor
	**/
	virtual ~tod_sum();

	/**	\brief Adds an operation to the end of the sequence
	 **/
	void add_op(tod_additive<N> &op, double c);

	//@}


	//!	\name Implementation of tod_additive<N>
	//@{

	virtual void prefetch();
    virtual void perform(cpu_pool &cpus, bool zero, double c,
        dense_tensor_i<N, double> &t);

	//@}

};


template<size_t N>
tod_sum<N>::tod_sum(tod_additive<N> &op, double c) {

	m_lst.push_back(node(op, c));
}


template<size_t N>
tod_sum<N>::~tod_sum() {

	m_lst.clear();
}


template<size_t N>
void tod_sum<N>::add_op(tod_additive<N> &op, double c) {

	m_lst.push_back(node(op, c));
}


template<size_t N>
void tod_sum<N>::prefetch() {

	for(typename std::list<node>::iterator i = m_lst.begin();
		i != m_lst.end(); i++) {

		i->m_op.prefetch();
	}
}


template<size_t N>
void tod_sum<N>::perform(cpu_pool &cpus, bool zero, double c,
    dense_tensor_i<N, double> &t) {

    if(zero) tod_set<N>().perform(cpus, t);

    for(typename std::list<node>::iterator i = m_lst.begin();
        i != m_lst.end(); ++i) {

        i->m_op.perform(cpus, false, c * i->m_c, t);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_SUM_H
