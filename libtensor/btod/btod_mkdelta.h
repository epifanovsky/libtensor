#ifndef LIBTENSOR_BTOD_MKDELTA_H
#define LIBTENSOR_BTOD_MKDELTA_H

#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/direct_block_tensor_operation.h"

namespace libtensor {


/**	\brief Forms the delta matrix \f$ \Delta_{ia} = f_{ii} - f_{aa} \f$

	\ingroup libtensor_btod
 **/
class btod_mkdelta :
	public direct_block_tensor_operation<2, double>,
	public timings<btod_mkdelta> {

public:
	static const char *k_clazz; //!< Class name

private:
	block_tensor_i<2, double> &m_fi; //!< Input tensor \f$ f_{ij} \f$
	block_tensor_i<2, double> &m_fa; //!< Input tensor \f$ f_{ab} \f$
	block_index_space<2> m_bis; //!< Block index space of result

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the operation
		\param fi Input tensor \f$ f_{ij} \f$.
		\param fa Input tensor \f$ f_{ab} \f$
	 **/
	btod_mkdelta(
		block_tensor_i<2, double> &fi, block_tensor_i<2, double> &fa);

	/**	\brief Virtual destructor
	 **/
	virtual ~btod_mkdelta() { }

	//@}


	//!	\name Implementation of
	//!		libtensor::direct_block_tensor_operation<2, double>
	//@{

	virtual const block_index_space<2> &get_bis() const;
	virtual const symmetry<2, double> &get_symmetry() const;
	virtual void perform(block_tensor_i<2, double> &bt) throw(exception);
	virtual void perform(block_tensor_i<2, double> &bt, const index<2> &i)
		throw(exception);

	//@}

private:
	block_index_space<2> mkbis(block_tensor_i<2, double> &fi,
		block_tensor_i<2, double> &fa);
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MKDELTA_H
