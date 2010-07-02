#ifndef LIBTENSOR_BTOD_READ_H
#define LIBTENSOR_BTOD_READ_H

#include <istream>
#include <sstream>
#include <libvmm/std_allocator.h>
#include "../defs.h"
#include "../exception.h"
#include "../timings.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../core/tensor_i.h"
#include "../core/tensor_ctrl.h"
#include "../tod/tod_compare.h"
#include "../tod/tod_set.h"
#include "../symmetry/so_copy.h"
#include "../symmetry/bad_symmetry.h"
#include "../btod/transf_double.h"

namespace libtensor {


/**	\brief Reads block %tensors from an input stream
	\tparam N Tensor order.
	\tparam Allocator Allocator for temporary tensors.

	The operation fills a block %tensor with data read from a formatted
	text input stream. Items in the stream are separated by whitespace
	characters. The format does not treat the new line character in any
	special way, it is another whitespace character.

	The first item in the stream is an integer specifying the order of
	the %tensor followed by a series of integers that specify the number of
	elements along each dimension of the %tensor. Then follow the actual
	data, each %tensor element is a double precision floating point number.

	After reading the data from the stream, the operation looks for zero
	blocks by checking that all elements are zero within a threshold
	(default 0.0 meaning that the elements must be exactly zero).

	The %symmetry of the block %tensor is guessed from the initial %symmetry
	set by the user before calling the operation. It is verified that
	the data actually have the specified %symmetry, otherwise an exception
	is raised. The comparison is done using a %symmetry threshold, within
	which two related elements are considered equal. The default value
	for the threshold is 0.0 meaning that the elements must be equal
	exactly.

	Format of the input stream:
	\code
	N D1 D2 ... Dn
	A1 A2 A3 ...
	\endcode
	N -- number of dimensions (integer); D1, D2, ..., Dn -- size of the
	%tensor along each dimension (N integers); A1, A2, A3, ... --
	%tensor elements (doubles).

	Example of a 3 by 3 antisymmetric matrix:
	\code
	2 3 3
	0.1 0.0 2.0
	0.0 1.3 -0.1
	-2.0 0.1 5.1
	\endcode

	\ingroup libtensor_btod
 **/
template<size_t N, typename Allocator = libvmm::std_allocator<double> >
class btod_read : public timings< btod_read<N, Allocator> > {
public:
	static const char *k_clazz; //!< Class name

private:
	std::istream &m_stream; //!< Input stream
	double m_zero_thresh; //!< Zero threshold
	double m_sym_thresh; //!< Symmetry threshold

public:
	//!	\name Construction and destruction
	//@{

	btod_read(std::istream &stream, double zero_thresh, double sym_thresh) :
		m_stream(stream), m_zero_thresh(zero_thresh),
		m_sym_thresh(sym_thresh) { }

	btod_read(std::istream &stream, double thresh = 0.0) :
		m_stream(stream), m_zero_thresh(thresh), m_sym_thresh(thresh)
		{ }

	//@}

	//!	\name Operation
	//@{

	void perform(block_tensor_i<N, double> &bt);

	//@}

private:
	void verify_zero_orbit(block_tensor_ctrl<N, double> &ctrl,
		const dimensions<N> &bidims, orbit<N, double> &o);

	void verify_nonzero_orbit(block_tensor_ctrl<N, double> &ctrl,
		const dimensions<N> &bidims, orbit<N, double> &o);

private:
	btod_read(const btod_read<N, Allocator>&);
	const btod_read<N, Allocator> &operator=(
		const btod_read<N, Allocator>&);

};


template<size_t N, typename Allocator>
const char *btod_read<N, Allocator>::k_clazz = "btod_read<N, Allocator>";


template<size_t N, typename Allocator>
void btod_read<N, Allocator>::perform(block_tensor_i<N, double> &bt) {

	static const char *method = "perform(block_tensor_i<N, double>&)";

	btod_read<N>::start_timer();

	//
	//	Read the first line: order, dimensions
	//

	if(!m_stream.good()) {
		throw bad_parameter(g_ns, k_clazz, method,
			__FILE__, __LINE__, "stream");
	}

	int order;
	index<N> i1, i2;
	size_t k = 0;
	m_stream >> order;
	if(order != N) {
		throw_exc(k_clazz, method, "Incorrect tensor order.");
	}
	while(m_stream.good() && k < N) {
		int dim;
		m_stream >> dim;
		if(dim <= 0) {
			throw_exc(k_clazz, method,
				"Incorrect tensor dimension.");
		}
		i2[k] = dim - 1;
		k++;
	}
	if(k < N) {
		throw_exc(k_clazz, method, "Unexpected end of stream.");
	}

	const block_index_space<N> &bis = bt.get_bis();
	dimensions<N> dims(index_range<N>(i1, i2));
	dimensions<N> bidims(bis.get_block_index_dims());
	if(!dims.equals(bis.get_dims())) {
		throw_exc(k_clazz, method, "Incompatible tensor dimensions.");
	}

	//
	//	Set up the tensor
	//

	block_tensor_ctrl<N, double> ctrl(bt);
	symmetry<N, double> sym(bis);
	so_copy<N, double>(ctrl.req_const_symmetry()).perform(sym);
	ctrl.req_symmetry().clear();
	ctrl.req_zero_all_blocks();

	//
	//	Read tensor elements from file into buffer
	//

	double *buf = new double[dims.get_size()];

	for(size_t i = 0; i < dims.get_size(); i++) {
		if(!m_stream.good()) {
			throw_exc(k_clazz, method, "Unexpected end of stream.");
		}
		m_stream >> buf[i];
	}

	//
	//	Transfer data into the block tensor
	//

	abs_index<N> bi(bidims);
	do {
		tensor_i<N, double> &blk = ctrl.req_block(bi.get_index());
		bool zero = true;
		{
		tensor_ctrl<N, double> blk_ctrl(blk);
		const dimensions<N> &blk_dims = blk.get_dims();
		double *p = blk_ctrl.req_dataptr();

		index<N> blk_start_idx(bis.get_block_start(bi.get_index()));
		abs_index<N> blk_offs_aidx(blk_dims);
		size_t nj = blk_dims[N - 1];
		do {
			index<N> idx(blk_start_idx);
			const index<N> &offs(blk_offs_aidx.get_index());
			for(size_t i = 0; i < N; i++) idx[i] += offs[i];
			abs_index<N> aidx(idx, bis.get_dims());
			size_t blk_offs = blk_offs_aidx.get_abs_index();
			size_t buf_offs = aidx.get_abs_index();
#ifdef LIBTENSOR_DEBUG
			if(buf_offs + nj > dims.get_size()) {
				throw out_of_bounds(g_ns, k_clazz, method,
					__FILE__, __LINE__,
					"buf_offs");
			}
#endif // LIBTENSOR_DEBUG
			for(size_t j = 0; j < nj; j++) {
				register double d = buf[buf_offs + j];
				if(fabs(d) <= m_zero_thresh) d = 0.0;
				else zero = false;
				p[blk_offs + j] = d;
			}
			for(size_t j = 1; j < nj; j++) blk_offs_aidx.inc();
		} while(blk_offs_aidx.inc());

		blk_ctrl.ret_dataptr(p);
		}
		ctrl.ret_block(bi.get_index());
		if(zero) ctrl.req_zero_block(bi.get_index());

	} while(bi.inc());

	delete [] buf;

	//
	//	Verify the symmetry of the block tensor
	//
	orbit_list<N, double>  ol(sym);
	for(typename orbit_list<N, double>::iterator io = ol.begin();
		io != ol.end(); io++) {

		orbit<N, double> o(sym, ol.get_index(io));
		abs_index<N> aci(o.get_abs_canonical_index(), bidims);

		if(ctrl.req_is_zero_block(aci.get_index())) {
			verify_zero_orbit(ctrl, bidims, o);
		} else {
			verify_nonzero_orbit(ctrl, bidims, o);
		}
	}

	//
	//	Re-install the symmetry of the block tensor
	//
	so_copy<N, double>(sym).perform(ctrl.req_symmetry());

	btod_read<N>::stop_timer();
}


template<size_t N, typename Allocator>
void btod_read<N, Allocator>::verify_zero_orbit(
	block_tensor_ctrl<N, double> &ctrl, const dimensions<N> &bidims,
	orbit<N, double> &o) {

	static const char *method =
		"verify_zero_orbit(block_tensor_ctrl<N, double>&, "
		"const dimensions<N>&, orbit<N, double>&)";

	typedef typename orbit<N, double>::iterator iterator_t;

	for(iterator_t i = o.begin(); i != o.end(); i++) {

		//	Skip the canonical block
		if(o.get_abs_index(i) == o.get_abs_canonical_index()) continue;

		//	Make sure the block is strictly zero
		abs_index<N> ai(o.get_abs_index(i), bidims);
		if(!ctrl.req_is_zero_block(ai.get_index())) {
			abs_index<N> aci(o.get_abs_canonical_index(), bidims);
			std::ostringstream ss;
			ss << "Asymmetry in zero block " << aci.get_index()
				<< "->" << ai.get_index() << ".";
			throw bad_symmetry(g_ns, k_clazz, method,
				__FILE__, __LINE__, ss.str().c_str());
		}
	}
}


template<size_t N, typename Allocator>
void btod_read<N, Allocator>::verify_nonzero_orbit(
	block_tensor_ctrl<N, double> &ctrl, const dimensions<N> &bidims,
	orbit<N, double> &o) {

	static const char *method =
		"verify_nonzero_orbit(block_tensor_ctrl<N, double>&, "
		"const dimensions<N>&, orbit<N, double>&)";

	typedef typename orbit<N, double>::iterator iterator_t;

	//	Get the canonical block
	abs_index<N> aci(o.get_abs_canonical_index(), bidims);
	tensor_i<N, double> &cblk = ctrl.req_block(aci.get_index());

	for(iterator_t i = o.begin(); i != o.end(); i++) {

		//	Skip the canonical block
		if(o.get_abs_index(i) == o.get_abs_canonical_index()) continue;

		//	Current index and transformation
		abs_index<N> ai(o.get_abs_index(i), bidims);
		const transf<N, double> &tr = o.get_transf(i);

		//	Compare with the transformed canonical block
		tensor_i<N, double> &blk = ctrl.req_block(ai.get_index());
		tensor<N, double, Allocator> tblk(blk.get_dims());
		tod_copy<N>(cblk, tr.get_perm(), tr.get_coeff()).perform(tblk);

		tod_compare<N> cmp(blk, tblk, m_sym_thresh);
		if(!cmp.compare()) {

			ctrl.ret_block(ai.get_index());
			ctrl.ret_block(aci.get_index());

			std::ostringstream ss;
			ss << "Asymmetry in block " << aci.get_index() << "->"
				<< ai.get_index() << " at element "
				<< cmp.get_diff_index() << ": "
				<< cmp.get_diff_elem_2() << " (expected), "
				<< cmp.get_diff_elem_1() << " (found), "
				<< cmp.get_diff_elem_1() - cmp.get_diff_elem_2()
				<< " (diff).";
			throw bad_symmetry(g_ns, k_clazz, method,
				__FILE__, __LINE__, ss.str().c_str());
		}

		ctrl.ret_block(ai.get_index());

		//	Zero out the block with proper symmetry
		ctrl.req_zero_block(ai.get_index());
	}

	ctrl.ret_block(aci.get_index());
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_READ_H
