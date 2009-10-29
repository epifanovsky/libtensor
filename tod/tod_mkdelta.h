#ifndef LIBTENSOR_TOD_MKDELTA_H
#define LIBTENSOR_TOD_MKDELTA_H

#include "defs.h"
#include "exception.h"
#include "timings.h"
#include "core/direct_tensor_operation.h"
#include "core/tensor_i.h"
#include "core/tensor_ctrl.h"

namespace libtensor {


/**	\brief Creates the delta matrix \f$ \Delta_{ia} = f_{ii}-f_{aa} \f$

	This operation forms the delta matrix from two diagonals:
	\f[ \Delta_{ia} = f_{ii} - f_{aa} \f]

	All the input and output tensors must agree in all the dimensions.

	The input matrices are provided while constructing the operation.
	In the constructor, their dimensions are checked. The resulting
	tensor is provided upon executing the operation, when its dimensions
	are also verified.

	Example:
	\code
	tensor_i<2, double> &foo(...), &fvv(...); // Input matrices
	tensor_i<2, double> &dov(...); // Delta matrix

	tod_mkdelta op(foo, fvv);
	op.perform(dov);
	\endcode

	\ingroup libtensor
 **/
class tod_mkdelta :
	public direct_tensor_operation<2, double>,
	public timings<tod_mkdelta> {

public:
	static const char *k_clazz; //!< Class name

private:
	tensor_i<2, double> &m_fi; //!< Input matrix \f$ f_{ij} \f$
	tensor_i<2, double> &m_fa; //!< Input matrix \f$ f_{ab} \f$

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the operation
		\param fi Input matrix \f$ f_{ij} \f$
		\param fa Input matrix \f$ f_{ab} \f$
	 **/
	tod_mkdelta(tensor_i<2, double> &fi, tensor_i<2, double> &fa);

	/**	\brief Virtual destructor
	 **/
	virtual ~tod_mkdelta();

	//@}


	//!	\name Implementation of
	//!		libtensor::direct_tensor_operation<2, double>
	//@{

	virtual void prefetch() throw(exception);
	virtual void perform(tensor_i<2, double> &delta) throw(exception);

	//@}
};

} // namespace libtensor

#endif // LIBTENSOR_TOD_MKDELTA_H

