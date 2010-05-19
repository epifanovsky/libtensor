#ifndef LIBTENSOR_TOD_ADD_H
#define LIBTENSOR_TOD_ADD_H

#include <list>
#include "tod_copy.h"
#include "bad_dimensions.h"

namespace libtensor {


/**	\brief Adds a series of tensors

	Tensor addition of n tensors:
	\f[
		B = \left( c_1 \mathcal{P}_1 A_1 + c_2 \mathcal{P}_2 A_2 + \cdots +
		c_n \mathcal{P}_n A_n \right) \f]

	Each operand must have the same dimensions as the result in order
	for the operation to be successful.

	\ingroup libtensor_tod
 **/
template<size_t N>
class tod_add : public timings< tod_add<N> > {
public:
	static const char* k_clazz; //!< Class name

private:
	struct arg {
		tensor_i<N, double> &t;
		permutation<N> p;
		double c;
		arg(tensor_i<N, double> &t_, const permutation<N> &p_,
			double c_ = 1.0) : t(t_), p(p_), c(c_) { }
	};

private:
	std::list<arg> m_args; //!< List of all operands to add
	dimensions<N> m_dims;  //!< Dimensions of the output tensor

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the addition operation
		\param t First %tensor in the series.
		\param c Scaling coefficient.
	 **/
	tod_add(tensor_i<N, double> &t, double c = 1.0);

	/**	\brief Initializes the addition operation
		\param t First %tensor in the series.
		\param p Permutation of the first %tensor.
		\param c Scaling coefficient.
	 **/
	tod_add(tensor_i<N, double> &t, const permutation<N> &p,
		double c = 1.0);

	/**	\brief Virtual destructor
	 **/
	virtual ~tod_add();

	//@}


	//!	\name Operation
	//@{

	/**	\brief Adds an operand
		\param t Tensor.
		\param c Coefficient.
	 **/
	void add_op(tensor_i<N, double> &t, double c);

	/**	\brief Adds an operand
		\param t Tensor.
		\param p Permutation of %tensor elements.
		\param c Coefficient.
	 **/
	void add_op(tensor_i<N, double> &t, const permutation<N> &p, double c);

	/**	\brief Prefetches the arguments
	 **/
	void prefetch();

	/**	\brief Computes the sum into the output %tensor
	 **/
	void perform(tensor_i<N, double> &t);

	/**	\brief Adds the sum to the output %tensor
	 **/
	void perform(tensor_i<N, double> &t, double c);

	//@}

private:
	/**	\brief Adds an operand (implementation)
	 **/
	void add_operand(tensor_i<N, double> &t, const permutation<N> &perm,
		double c);

};


template<size_t N>
const char* tod_add<N>::k_clazz = "tod_add<N>";


template<size_t N>
tod_add<N>::tod_add(tensor_i<N, double> &t, double c) : m_dims(t.get_dims()) {

	static const char *method = "tod_add(tensor_i<N, double>&, double)";

	add_operand(t, permutation<N>(), c);
}


template<size_t N>
tod_add<N>::tod_add(tensor_i<N, double> &t, const permutation<N> &p, double c) :
	m_dims(t.get_dims()) {

	static const char *method =
		"tod_add(tensor_i<N, double>&, const permutation<N>&, double)";

	m_dims.permute(p);
	add_operand(t, p, c);
}


template<size_t N>
tod_add<N>::~tod_add() {

}


template<size_t N>
void tod_add<N>::add_op(tensor_i<N, double> &t, double c) {

	static const char *method = "add_op(tensor_i<N, double>&, double)";

	if(c == 0.0) return;

	if(!t.get_dims().equals(m_dims)) {
		throw bad_dimensions(
			g_ns, k_clazz, method, __FILE__, __LINE__, "t");
	}

	add_operand(t, permutation<N>(), c);
}


template<size_t N>
void tod_add<N>::add_op(
	tensor_i<N, double> &t, const permutation<N> &p, double c) {

	static const char *method =
		"add_op(tensor_i<N, double>&, const permutation<N>&, double)";

	if(c == 0.0) return;

	dimensions<N> dims(t.get_dims());
	dims.permute(p);
	if(!dims.equals(m_dims)) {
		throw bad_dimensions(
			g_ns, k_clazz, method, __FILE__, __LINE__, "t");
	}

	add_operand(t, p, c);
}


template<size_t N>
void tod_add<N>::add_operand(tensor_i<N, double> &t, const permutation<N> &p,
	double c) {

	static const char *method = "add_operand(tensor_i<N, double>&, "
		"const permutation<N>&, double)";

	m_args.push_back(arg(t, p, c));
}


template<size_t N>
void tod_add<N>::prefetch() {

	for(typename std::list<arg>::iterator i = m_args.begin();
		i != m_args.end(); i++) {

		tensor_ctrl<N, double>(i->t).req_prefetch();
	}
}


template<size_t N>
void tod_add<N>::perform(tensor_i<N, double> &t) {

	static const char *method = "perform(tensor_i<N, double>&)";

	//	Check the dimensions of the output tensor
	if(!t.get_dims().equals(m_dims)) {
		throw bad_dimensions(
			g_ns, k_clazz, method, __FILE__, __LINE__, "t");
	}

	tod_add<N>::start_timer();

	typename std::list<arg>::iterator i = m_args.begin();
	tod_copy<N>(i->t, i->p, i->c).perform(t);
	i++;
	for(; i != m_args.end(); i++) {
		tod_copy<N>(i->t, i->p, i->c).perform(t, 1.0);
	}

	tod_add<N>::stop_timer();
}


template<size_t N>
void tod_add<N>::perform(tensor_i<N, double> &t, double c) {

	static const char *method = "perform(tensor_i<N, double>&, double)";

	//	Check the dimensions of the output tensor
	if(!t.get_dims().equals(m_dims)) {
		throw bad_dimensions(
			g_ns, k_clazz, method, __FILE__, __LINE__, "t");
	}

	if(c == 0.0) return;

	tod_add<N>::start_timer();

	typename std::list<arg>::iterator i = m_args.begin();
	for(; i != m_args.end(); i++) {
		tod_copy<N>(i->t, i->p, i->c).perform(t, c);
	}

	tod_add<N>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_ADD_H

