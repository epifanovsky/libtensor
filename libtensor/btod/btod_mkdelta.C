#include <libtensor/tod/tod_copy.h>
#include <libtensor/tod/tod_mkdelta.h>
#include "btod_mkdelta.h"


namespace libtensor {


const char *btod_mkdelta::k_clazz = "btod_mkdelta";


btod_mkdelta::btod_mkdelta(
	block_tensor_i<2, double> &fi, block_tensor_i<2, double> &fa) :
	m_fi(fi), m_fa(fa), m_bis(mkbis(fi, fa)) {


}


const block_index_space<2> &btod_mkdelta::get_bis() const {

	return m_bis;
}


const symmetry<2, double> &btod_mkdelta::get_symmetry() const {

	throw_exc(k_clazz, "get_symmetry()", "NIY");
}


void btod_mkdelta::perform(block_tensor_i<2, double> &bt) throw(exception) {

	static const char *method = "perform(block_tensor_i<2, double>&)";

	if(!m_bis.equals(bt.get_bis())) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"Incompatible block tensor index spaces.");
	}

	block_tensor_ctrl<2, double> ctrl_i(m_fi), ctrl_a(m_fa);
	block_tensor_ctrl<2, double> ctrl_d(bt);
	dimensions<2> bidims(m_bis.get_block_index_dims());
	size_t ni = bidims[0], na = bidims[1];
	// Temporary way to deal with alpha-beta; to be replaced with
	// appropriate symmetry.
	size_t ni_ab = ni / 2, na_ab = na / 2;

	ctrl_d.req_sym_clear_elements();

	index<2> idx_i, idx_a, idx_d;
	for(size_t i = 0; i < ni; i++) {
		for(size_t a = 0; a < na; a++) {

			idx_i[0] = i; idx_i[1] = i;
			idx_a[0] = a; idx_a[1] = a;
			idx_d[0] = i; idx_d[1] = a;

			if((i < ni_ab) != (a < na_ab)) {
				ctrl_d.req_zero_block(idx_d);
				continue;
			}

			if(ctrl_i.req_is_zero_block(idx_i)) {
				if(ctrl_a.req_is_zero_block(idx_a)) {
					ctrl_d.req_zero_block(idx_d);
				} else {
					tensor_i<2, double> &blk_a =
						ctrl_a.req_block(idx_a);
					tensor_i<2, double> &blk_d =
						ctrl_d.req_block(idx_d);
					tod_copy<2>(blk_a, -1.0).perform(blk_d);
					ctrl_a.ret_block(idx_a);
					ctrl_d.ret_block(idx_d);
				}
			} else {
				if(ctrl_a.req_is_zero_block(idx_a)) {
					tensor_i<2, double> &blk_i =
						ctrl_i.req_block(idx_i);
					tensor_i<2, double> &blk_d =
						ctrl_d.req_block(idx_d);
					tod_copy<2>(blk_i, 1.0).perform(blk_d);
					ctrl_i.ret_block(idx_i);
					ctrl_d.ret_block(idx_d);
				} else {
					tensor_i<2, double> &blk_i =
						ctrl_i.req_block(idx_i);
					tensor_i<2, double> &blk_a =
						ctrl_a.req_block(idx_a);
					tensor_i<2, double> &blk_d =
						ctrl_d.req_block(idx_d);
					tod_mkdelta(blk_i, blk_a).
						perform(blk_d);
					ctrl_i.ret_block(idx_i);
					ctrl_a.ret_block(idx_a);
					ctrl_d.ret_block(idx_d);
				}
			}
		}
	}
}


void btod_mkdelta::perform(block_tensor_i<2, double> &bt, const index<2> &i)
	throw(exception) {

	throw_exc(k_clazz, "perform(index)", "NIY");
}


block_index_space<2> btod_mkdelta::mkbis(block_tensor_i<2, double> &fi,
	block_tensor_i<2, double> &fa) {

	static const char *method = "btod_mkdelta::mkbis()";

	const block_index_space<2> &bisi = fi.get_bis();
	const block_index_space<2> &bisa = fa.get_bis();

	if(bisi.get_type(0) != bisi.get_type(1)) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"fi");
	}
	if(bisa.get_type(0) != bisa.get_type(1)) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"fa");
	}

	index<2> i1, i2;
	i2[0] = bisi.get_dims()[0] - 1;
	i2[1] = bisa.get_dims()[0] - 1;
	dimensions<2> dims(index_range<2>(i1, i2));
	block_index_space<2> bis(dims);

	mask<2> mski, mska;
	mski[0] = true; mska[1] = true;

	const split_points &ptsi = bisi.get_splits(bisi.get_type(0));
	size_t ni = ptsi.get_num_points();
	for(size_t i = 0; i < ni; i++) {
		bis.split(mski, ptsi[i]);
	}
	const split_points &ptsa = bisa.get_splits(bisa.get_type(0));
	size_t na = ptsa.get_num_points();
	for(size_t a = 0; a < na; a++) {
		bis.split(mska, ptsa[a]);
	}

	return bis;
}


} // namespace libtensor
