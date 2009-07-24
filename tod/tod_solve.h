#ifndef LIBTENSOR_TOD_DIAG_H
#define LIBTENSOR_TOD_DIAG_H

#include "defs.h"
#include "exception.h"
#include "core/tensor_i.h"
#include "core/tensor_ctrl.h"
#include "core/direct_tensor_operation.h"
#include "core/dimensions.h"

namespace libtensor {

	/**	\brief The routine solves for X the system of linear equations A*X = B,
	 * 		   where A is an n-by-n matrix, the vector B are individual
	 * 		   right-hand side, and the column of X are the corresponding solution.


		\ingroup libtensor_tod
	**/
	//template<size_t N>
	class tod_solve {
	private:
		tensor_i<2,double> &m_tmatrix;
		tensor_i<1,double> &m_tvect;
		tensor_ctrl<2,double> m_tctrl_matrix;
		tensor_ctrl<1,double> m_tctrl_vect;

	public:
		//!	\name Construction and destruction
		//@{

		/**	\brief Initializes the operation
			\param tmatrix Tensor contains the n-by-n coefficient matrix A.
			\param tvect Tensor contains contains the n vector of right hand side vector B.

		**/
		tod_solve(tensor_i<2,double> &tmatrix, tensor_i<1,double> &tvect);

		/**	\brief Virtual destructor
		**/
		virtual ~tod_solve();

		//@}

		//don't use right meanwhile
		//!	\name Implementation of direct_tensor_operation
		//@{


		/**	\brief Solve the equation
			\throw Exception if the tensors have different dimensions
				or another error occurs
		**/
		virtual void perform(tensor_i<1,double> &result_vect) throw(exception);

		//@}

	};

} // namespace libtensor

#endif // LIBTENSOR_TOD_SOLVE_H

