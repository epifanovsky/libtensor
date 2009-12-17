#ifndef LIBTENSOR_BTOD_IMPORT_RAW_H
#define LIBTENSOR_BTOD_IMPORT_RAW_H

#include "defs.h"
#include "exception.h"
#include "core/abs_index.h"
#include "core/dimensions.h"
#include "core/block_tensor_i.h"
#include "tod/tod_import_raw.h"

namespace libtensor {


/**	\brief Imports block %tensor elements from memory
	\tparam N Tensor order.

	This operation reads %tensor elements from a memory block in which
	they are arranged in the regular %tensor format.

	\sa tod_import_raw<N>

	\ingroup libtensor_btod
 **/
template<size_t N>
class btod_import_raw {
public:
	static const char *k_clazz; //!< Class name

private:
	const double *m_ptr; //!< Pointer to data in memory
	dimensions<N> m_dims; //!< Dimensions of the memory block

public:
	/**	\brief Initializes the operation
		\param ptr Memory pointer.
		\param dims Dimensions of the input.
	 **/
	btod_import_raw(const double *ptr, const dimensions<N> &dims) :
		m_ptr(ptr), m_dims(dims) { }

	/**	\brief Performs the operation
		\param bt Output block %tensor.
	 **/
	void perform(block_tensor_i<N, double> &bt);
};


template<size_t N>
const char *btod_import_raw<N>::k_clazz = "btod_import_raw<N>";


template<size_t N>
void btod_import_raw<N>::perform(block_tensor_i<N, double> &bt) {

	static const char *method = "perform(block_tensor_i<N>&)";

	//	Check the block tensor's dimensions

	const block_index_space<N> &bis = bt.get_bis();
	if(!bis.get_dims().equals(m_dims)) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Incorrect block tensor dimensions.");
	}

	//	Set up the block tensor

	block_tensor_ctrl<N, double> ctrl(bt);
	ctrl.req_sym_clear_elements();
	ctrl.req_zero_all_blocks();

	//	Invoke the import operation for each block

	dimensions<N> bdims(bis.get_block_index_dims());
	abs_index<N> bi(bdims);

	do {

		tensor_i<N, double> &blk = ctrl.req_block(bi.get_index());

		index<N> blk_start(bis.get_block_start(bi.get_index()));
		dimensions<N> blk_dims(bis.get_block_dims(bi.get_index()));
		index<N> blk_end(blk_start);
		for(size_t i = 0; i < N; i++) blk_end[i] += blk_dims[i] - 1;
		index_range<N> ir(blk_start, blk_end);

		tod_import_raw<N>(m_ptr, m_dims, ir).perform(blk);

		ctrl.ret_block(bi.get_index());

	} while(bi.inc());
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_IMPORT_RAW_H
