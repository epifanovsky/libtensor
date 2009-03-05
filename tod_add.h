#ifndef LIBTENSOR_TOD_ADD_H
#define LIBTENSOR_TOD_ADD_H

#include "defs.h"
#include "exception.h"
#include "tensor_i.h"
#include "direct_tensor_additive_operation.h"

namespace libtensor {

/**	\brief Adds two or more tensors

	Tensor addition of n tensors:
	\f[ B = c_1 \mathcal{P}_1 A_1 + c_2 \mathcal{P}_2 A_2 + \cdots +
		c_n \mathcal{P}_n A_n \f]

	Each operand must have the same dimensions as the result in order
	for the operation to be successful. 

	

	\ingroup libtensor_tod
**/
class tod_add : public direct_tensor_additive_operation<double> {
private:
	struct operand {
		tensor_i<double> &m_t;
		const permutation &m_p;
		const double m_c;
		operand(tensor_i<double> &t, const permutation &p,
			const double m_c) : m_t(t), m_p(p), m_c(c) {}
	};

	tensor_i<double> &m_out; //!< Output tensor
	permutation &m_perm_out; //!< Output permutation
	bool m_add; //!< Add/replace output
	dimensions m_dims_out; //!< Permuted dimensions of the output

public:
	/**	\brief Initializes the operation
		\param t Output tensor.
		\param p Output permutation.
		\param add Whether the result should be added to the tensor
			or replace the current contents.
	**/
	tod_add(tensor_i<double> &t, const permutation &p, const bool add)
		throw(exception);

	/**	\brief Destroys the operation
	**/
	~tod_add();

	/**	\brief Adds an operand
		\param t Tensor.
		\param p Permutation of %tensor elements.
		\param c Coefficient.
	**/
	void add_op(tensor_i<double> &t, const permutation &p, const double c)
		throw(exception);

	//!	\name Implementation of direct_tensor_operation<T>
	//@{
	void perform(tensor_i<double> &t) throw(exception);
	//@}

	//!	\name Implementation of direct_tensor_additive_operation<T>
	//@{
	void perform(tensor_i<double> &t, const double c) throw(exception);
	//@}
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_ADD_H

