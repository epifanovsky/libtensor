#ifndef LIBTENSOR_BTOD_IMPORT_RAW_H
#define LIBTENSOR_BTOD_IMPORT_RAW_H

#include "../defs.h"
#include "../exception.h"
#include "../core/abs_index.h"
#include "../core/dimensions.h"
#include "../core/block_tensor_i.h"
#include "../core/orbit.h"
#include "../symmetry/bad_symmetry.h"
#include "../symmetry/so_copy.h"
#include "../tod/tod_compare.h"
#include "../tod/tod_copy.h"
#include "../tod/tod_import_raw.h"

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
	symmetry<N, double> sym(bis);
	so_copy<N, double>(ctrl.req_const_symmetry()).perform(sym);
	ctrl.req_symmetry().clear();
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


	//
	// Check block tensor for its symmetry
	//

	orbit_list<N, double>  ol(sym);
	// loop over all orbits
	for (typename orbit_list<N, double>::iterator it = ol.begin();
			it != ol.end(); it++) {

		orbit<N, double> orb(sym, ol.get_index(it));

		// get canonical block
		size_t ac = orb.get_abs_canonical_index();
		abs_index<N> acidx(ac, bdims);

		bool zero = ctrl.req_is_zero_block(acidx.get_index());

		if (zero) {
			// loop over all blocks within an orbit
			for (typename orbit<N, double>::iterator ito = orb.begin();
					ito != orb.end(); ito++) {

				// do nothing for the canonical block
				if (ac == orb.get_abs_index(ito)) continue;

				abs_index<N> aidx(orb.get_abs_index(ito), bdims);

				if (ctrl.req_is_zero_block(aidx.get_index())) continue;

				std::ostringstream oss;
				oss << "Read block tensor does not match symmetry in block "
						<< aidx.get_index() << "(non-zero block).";
				throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
						oss.str().c_str());
			}
		}
		else {
			// apply transformation to the canonical block
			tensor_i<N, double> &cblk = ctrl.req_block(acidx.get_index());

			// loop over all blocks within an orbit
			for (typename orbit<N, double>::iterator ito = orb.begin();
					ito != orb.end(); ito++) {

				// do nothing for the canonical block
				if (ac == orb.get_abs_index(ito)) continue;

				// get block transformation and index
				abs_index<N> aidx(orb.get_abs_index(ito), bdims);
				const transf<N, double> &tr = orb.get_transf(ito);

				// compare real and expected block
				tensor_i<N, double> &blk = ctrl.req_block(aidx.get_index());

				typedef libvmm::std_allocator<double> allocator;
				tensor<N, double, allocator> tmp_blk(cblk.get_dims());
				tod_copy<N>(cblk, tr.get_perm(), tr.get_coeff()).perform(tmp_blk);

				tod_compare<N> cmp(cblk, tmp_blk, 1e-13);
				if (! cmp.compare()) {
					ctrl.ret_block(aidx.get_index());
					ctrl.ret_block(acidx.get_index());

					std::ostringstream oss;
					oss << "Read block tensor does not match symmetry in block "
							<< aidx.get_index() << " at " << cmp.get_diff_index()
							<< "(expected: " << cmp.get_diff_elem_1() << ", found: "
							<< cmp.get_diff_elem_2() << ", diff: "
							<< cmp.get_diff_elem_1()-cmp.get_diff_elem_2() << ").";

					throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
							oss.str().c_str());
				}

				ctrl.ret_block(aidx.get_index());
				// if real and expected block match zero the non-canonical block
				ctrl.req_zero_block(aidx.get_index());
			}
			ctrl.ret_block(acidx.get_index());

		}
	}

	// copy symmetry back to block tensor
	so_copy<N, double>(sym).perform(ctrl.req_symmetry());

}

} // namespace libtensor

#endif // LIBTENSOR_BTOD_IMPORT_RAW_H
