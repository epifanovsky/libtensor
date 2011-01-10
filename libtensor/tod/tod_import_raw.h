#ifndef LIBTENSOR_TOD_IMPORT_RAW_H
#define LIBTENSOR_TOD_IMPORT_RAW_H

#include "../defs.h"
#include "../exception.h"
#include "../linalg/linalg.h"
#include "../core/dimensions.h"
#include "../core/index_range.h"
#include "processor.h"
#include "bad_dimensions.h"

namespace libtensor {


/**	\brief Imports %tensor elements from memory
	\tparam N Tensor order.

	This operation reads %tensor elements from a given window of a block
	of memory. The elements in the memory must be in the usual %tensor
	format. The block is characterized by its %dimensions, as if it were
	a part of the usual %tensor object. The window is specified by a range
	of indexes.

	The size of the recipient (result of the operation) must agree with
	the window dimensions.

	\ingroup libtensor_tod
 **/
template<size_t N>
class tod_import_raw {
public:
	static const char *k_clazz; //!< Class name

private:
	const double *m_ptr; //!< Pointer to data
	dimensions<N> m_dims; //!< Dimensions of the memory block
	index_range<N> m_ir; //!< Index range of the window

private:
	struct registers {
		const double *m_ptra;
		double *m_ptrb;
	};

	struct loop_list_node;
	typedef std::list<loop_list_node> loop_list_t;
	typedef processor<loop_list_t, registers> processor_t;
	typedef processor_op_i<loop_list_t, registers> processor_op_i_t;

	struct loop_list_node {
	public:
		size_t m_weight;
		size_t m_inca, m_incb;
		processor_op_i_t *m_op;
		loop_list_node()
			: m_weight(0), m_inca(0), m_incb(0), m_op(NULL) { }
		loop_list_node(size_t weight, size_t inca, size_t incb)
			: m_weight(weight), m_inca(inca), m_incb(incb),
			m_op(NULL) { }
		processor_op_i_t *op() const { return m_op; }
	};

	class op_loop : public processor_op_i_t {
	private:
		size_t m_len, m_inca, m_incb;
	public:
		op_loop(size_t len, size_t inca, size_t incb)
			: m_len(len), m_inca(inca), m_incb(incb) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	class op_dcopy : public processor_op_i_t {
	private:
		size_t m_len;
	public:
		op_dcopy(size_t len) : m_len(len) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

public:
	/**	\brief Initializes the operation
		\param ptr Pointer to data block
		\param dims Dimensions of the data block
		\param ir Index range of the window
	 **/
	tod_import_raw(const double *ptr, const dimensions<N> &dims,
		const index_range<N> &ir) :
			m_ptr(ptr), m_dims(dims), m_ir(ir) { }

	/**	\brief Performs the operation
		\param t Output %tensor
	 **/
	void perform(tensor_i<N, double> &t);

};


template<size_t N>
const char *tod_import_raw<N>::k_clazz = "tod_import_raw<N>";


template<size_t N>
void tod_import_raw<N>::perform(tensor_i<N, double> &t) {

	static const char *method = "perform(tensor_i<N, double>&)";

	dimensions<N> dimsb(m_ir);
	if(!t.get_dims().equals(dimsb)) {
		throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t.");
	}

	tensor_ctrl<N, double> tctrl(t);

	loop_list_t lst;
	for(size_t i = 0; i < N; i++) {
		size_t inca = m_dims.get_increment(i);
		size_t incb = dimsb.get_increment(i);
		loop_list_node node(dimsb[i], inca, incb);
		if(i == N - 1) {
			node.m_op = new op_dcopy(dimsb[i]);
		} else {
			node.m_op = new op_loop(dimsb[i], inca, incb);
		}
		lst.push_back(node);
	}

	const double *ptra = m_ptr +
		abs_index<N>(m_ir.get_begin(), m_dims).get_abs_index();
	double *ptrb = tctrl.req_dataptr();

	registers regs;
	regs.m_ptra = ptra;
	regs.m_ptrb = ptrb;
	processor_t proc(lst, regs);
	proc.process_next();

	tctrl.ret_dataptr(ptrb);

	for(typename loop_list_t::iterator i = lst.begin();
		i != lst.end(); i++) {

		delete i->m_op;
		i->m_op = NULL;
	}
}


template<size_t N>
void tod_import_raw<N>::op_loop::exec(processor_t &proc, registers &regs)
	throw(exception) {

	const double *ptra = regs.m_ptra;
	double *ptrb = regs.m_ptrb;

	for(size_t i=0; i<m_len; i++) {
		regs.m_ptra = ptra;
		regs.m_ptrb = ptrb;
		proc.process_next();
		ptra += m_inca;
		ptrb += m_incb;
	}
}


template<size_t N>
void tod_import_raw<N>::op_dcopy::exec(processor_t &proc, registers &regs)
	throw(exception) {

	linalg::i_i(m_len, regs.m_ptra, 1, regs.m_ptrb, 1);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_IMPORT_RAW_H
