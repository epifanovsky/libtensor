#ifndef LIBTENSOR_TOD_ADD_H
#define LIBTENSOR_TOD_ADD_H

#include <list>
#include <map>

#include "../defs.h"
#include "../exception.h"
#include "../blas.h"
#include "../timings.h"
#include "../core/tensor_i.h"
#include "../core/tensor_ctrl.h"
#include "tod_additive.h"
#include "processor.h"
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
class tod_add : public tod_additive<N>, public timings< tod_add<N> > {
public:
	static const char* k_clazz; //!< Class name

private:
	typedef struct operand {
		tensor_i<N,double>& m_ta;
		double m_ca;
		operand( tensor_i<N,double>& ta, double ca ) : m_ta(ta), m_ca(ca) {}
	} operand_t;

	typedef std::pair<permutation<N>,operand_t*> op_pair_t;
	typedef std::multimap<permutation<N>,operand_t*> op_map_t;

	typedef struct {
		const double* m_ptra; //!< tensor to add
		double m_ca; //!< scaling factor to add
		double* m_ptrb; //!< result tensor
	} registers;

	struct loop_list_node;
	typedef std::list<loop_list_node> loop_list_t;
	typedef processor<loop_list_t, registers> processor_t;
	typedef processor_op_i<loop_list_t, registers> processor_op_i_t;

	struct loop_list_node {
		processor_op_i_t* m_op;
		size_t m_len, m_inca, m_incb;

		loop_list_node() : m_op(NULL) { }
		loop_list_node( size_t len, size_t inca, size_t incb ) :
			m_len(len), m_inca(inca), m_incb(incb), m_op(NULL) { }
		processor_op_i_t *op() const { return m_op; }
	};

	class op_loop : public processor_op_i_t {
	private:
		size_t m_len, m_inca, m_incb;
	public:
		op_loop(size_t len, size_t inca, size_t incb) :
			m_len(len), m_inca(inca), m_incb(incb) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);
	};

	//!	b_j += m_ca * a_{i_j}
	class op_daxpy : public processor_op_i_t, public timings<op_daxpy> {
	private:
		size_t m_len, m_inca, m_incb;
	public:
		op_daxpy(size_t len , size_t inca, size_t incb) :
			m_inca(inca), m_incb(incb), m_len(len) { }
		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);

		static const char* k_clazz;
	};

	//!	b_{ij} += m_ca * a_{ji}
	class op_daxpby_trp : public processor_op_i_t, public timings<op_daxpby_trp> {
	private:
		size_t m_leni, m_lenj, m_incaj, m_incbi;
	public:
		op_daxpby_trp(size_t leni , size_t lenj, size_t incaj, size_t incbi) :
			m_incaj(incaj), m_incbi(incbi), m_leni(leni), m_lenj(lenj) { }

		virtual void exec(processor_t &proc, registers &regs)
			throw(exception);

		static const char* k_clazz;
	};


	op_map_t m_operands; //!< List of all operands to add
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

	/**	\copydoc tod_additive<N>::prefetch()
	 **/
	virtual void prefetch() throw(exception);

	/**	\copydoc tod_additive<N>::perform(tensor_i<N, double>&)
	 **/
	virtual void perform(tensor_i<N,double> &t) throw(exception);

	/**	\copydoc tod_additive<N>::perform(tensor_i<N, double>&, double)
	 **/
	virtual void perform(tensor_i<N,double> &t, double c) throw(exception);

	//@}

private:
	/**	\brief Adds an operand (implementation)
	 **/
	void add_operand(
		tensor_i<N, double> &t, const permutation<N> &perm, double c);

	/**	\brief Build operations list for addition of two tensors

		\params list resulting list of operations
		\params da dimensions of %tensor to add
		\params pa permutation of %tensor to add
		\params db dimensions of %tensor to add to

		The resulting operations list is appropriate to add a tensor with
		dimensions da and permutation pa to tensor with dimensions db
	**/
	void build_list( loop_list_t &list, const dimensions<N> &da,
		const permutation<N> &pa, const dimensions<N> &db ) throw(out_of_memory);
	//!	Delete all operations in list
	void clean_list( loop_list_t &list );


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

	typename op_map_t::iterator i = m_operands.begin();
	for(; i != m_operands.end(); i++) {
		delete i->second;
	}
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
void tod_add<N>::add_operand(
	tensor_i<N, double> &t, const permutation<N> &p, double c) {

	static const char *method = "add_operand(tensor_i<N, double>&, "
		"const permutation<N>&, double)";

	try {
		operand_t *op = new operand_t(t, c);
		m_operands.insert(op_pair_t(p, op));
	} catch(std::bad_alloc& e) {
		throw out_of_memory(
			g_ns, k_clazz, method, __FILE__,__LINE__, e.what());
	}
}


template<size_t N>
void tod_add<N>::prefetch() throw(exception) {

	typename op_map_t::iterator i = m_operands.begin();
	for(; i != m_operands.end(); i++) {
		tensor_ctrl<N, double>(i->second->m_ta).req_prefetch();
	}
}


template<size_t N>
void tod_add<N>::perform(tensor_i<N, double> &t) throw(exception) {

	static const char *method = "perform(tensor_i<N, double>&)";

	//	Check the dimensions of the output tensor
	if(!t.get_dims().equals(m_dims)) {
		throw bad_dimensions(
			g_ns, k_clazz, method, __FILE__, __LINE__, "t");
	}

	tod_add<N>::start_timer();

	tensor_ctrl<N,double> ctrlt(t);
	double* tptr=ctrlt.req_dataptr();
	// set all elements of t to zero
	memset(tptr,0,m_dims.get_size()*sizeof(double));

	double* ptr=NULL;
	typename op_map_t::iterator it=m_operands.begin();
	while ( it != m_operands.end() ) {
		if ( m_operands.count(it->first) == 1 ) {
			tensor_ctrl<N,double> ctrlo(it->second->m_ta);
			const double* optr=ctrlo.req_const_dataptr();

			loop_list_t list;
			build_list(list,it->second->m_ta.get_dims(),it->first,m_dims);
			try {
				registers regs;
				regs.m_ptra=optr;
				regs.m_ca=it->second->m_ca;
				regs.m_ptrb=tptr;
				processor_t(list,regs).process_next();
			}
			catch ( exception& e ) {
				clean_list(list);
				throw;
			}
			clean_list(list);
			it++;
		}
		else {
			const permutation<N>& perm=it->first;
			dimensions<N> dim=it->second->m_ta.get_dims();

			try {
				if ( ptr == NULL ) ptr=new double[dim.get_size()];
			} catch ( std::bad_alloc& e ) {
				throw out_of_memory("libtensor",k_clazz,"perform(tensor_i<N,double> &)",
						__FILE__,__LINE__,e.what());
			}

			memset(ptr,0,m_dims.get_size()*sizeof(double));
			while ( it != m_operands.upper_bound(perm) ) {
				tensor_ctrl<N,double> ctrlo(it->second->m_ta);
				const double* optr=ctrlo.req_const_dataptr();
				cblas_daxpy(dim.get_size(),it->second->m_ca,optr,1,ptr,1);
				ctrlo.ret_dataptr(optr);
				it++;
			}

			loop_list_t list;
			build_list(list,dim,perm,m_dims);

			try {
				registers regs;
				regs.m_ptra=ptr;
				regs.m_ca=1.0;
				regs.m_ptrb=tptr;
				processor_t(list,regs).process_next();
			} catch ( exception& e ) {
				delete [] ptr;
				clean_list(list);
				throw;
			}
			clean_list(list);
		}
	}

	if ( ptr != NULL ) delete [] ptr;

	tod_add<N>::stop_timer();
}

template<size_t N>
void tod_add<N>::perform(tensor_i<N, double> &t, double c) throw(exception) {

	static const char *method = "perform(tensor_i<N, double>&, double)";

	//	Check the dimensions of the output tensor
	if(!t.get_dims().equals(m_dims)) {
		throw bad_dimensions(
			g_ns, k_clazz, method, __FILE__, __LINE__, "t");
	}

	if(c == 0.0) return;

	tod_add<N>::start_timer();

	tensor_ctrl<N,double> ctrlt(t);
	double* tptr=ctrlt.req_dataptr();

	double* ptr=NULL;

	try {

		typename op_map_t::iterator it=m_operands.begin();
		while ( it != m_operands.end() ) {
			if ( m_operands.count(it->first) == 1 ) {
				tensor_ctrl<N,double> ctrlo(it->second->m_ta);
				const double* optr=ctrlo.req_const_dataptr();

				loop_list_t list;
				build_list(list,it->second->m_ta.get_dims(),it->first,m_dims);
				try {
					registers regs;
					regs.m_ptra=optr;
					regs.m_ca=it->second->m_ca*c;
					regs.m_ptrb=tptr;
					processor_t(list,regs).process_next();
				}
				catch ( exception& e ) {
					clean_list(list);
					throw;
				}
				clean_list(list);
				it++;
			}
			else {
				const permutation<N>& perm=it->first;
				dimensions<N> dim=it->second->m_ta.get_dims();

				try {
					if ( ptr == NULL ) ptr=new double[dim.get_size()];
				} catch ( std::bad_alloc& e ) {
					throw out_of_memory("libtensor",k_clazz,
						"perform(tensor_i<N,double> &)",
						__FILE__,__LINE__,e.what());
				}

				memset(ptr,0,dim.get_size()*sizeof(double));

				while ( it != m_operands.upper_bound(perm) ) {
					tensor_ctrl<N,double> ctrlo(it->second->m_ta);
					const double* optr=ctrlo.req_const_dataptr();
					cblas_daxpy(dim.get_size(),it->second->m_ca,optr,1,ptr,1);
					ctrlo.ret_dataptr(optr);
					it++;
				}

				loop_list_t list;
				build_list(list,dim,perm,m_dims);

				try {
					registers regs;
					regs.m_ptra=ptr;
					regs.m_ca=c;
					regs.m_ptrb=tptr;
					processor_t(list,regs).process_next();
				} catch ( exception& e ) {
					clean_list(list);
					throw;
				}
				clean_list(list);
			}
		}

		if ( ptr != NULL ) delete [] ptr;
	} catch ( exception& e ) {
		if ( ptr != NULL ) delete [] ptr;
		throw;
	}

	tod_add<N>::stop_timer();
}

template<size_t N>
void tod_add<N>::build_list( loop_list_t &list, const dimensions<N> &da,
	const permutation<N> &pa, const dimensions<N> &db ) throw(out_of_memory)
{
	size_t ia[N];
	for (size_t i=0; i<N; i++) ia[i]=i;
	pa.apply(N,ia);

	// loop over all indices and build the list
	size_t pos=0;
	try {
		typename loop_list_t::iterator posa=list.end(), posb=list.end();
		while ( pos < N ) {
			size_t len=1;
			size_t iapos=ia[pos];
			while (pos<N) {
				len*=da.get_dim(iapos);
				pos++; iapos++;
				if ( ia[pos]!=iapos ) break;
			}

			size_t inca=da.get_increment(iapos-1);
			size_t incb=db.get_increment(pos-1);

			typename loop_list_t::iterator it
				= list.insert(list.end(),loop_list_node(len,inca,incb));

			// we know that in the last loop incb=1 !!!
			if (inca==1) {
				if (inca==incb)	{
					it->m_op=new op_daxpy(len,inca,incb);
				}
				else { posa=it; }
			}
			else {
				if (incb==1) { posb=it; }
				else { it->m_op=new op_loop(len,inca,incb); }
			}
		}

		if ( posa!=posb ) {
//			if ( posa->m_len > posb->m_len ) {
//				posa->m_op=new op_daxpy(posa->m_len,posa->m_inca,posa->m_incb);
//				posb->m_op=new op_loop(posb->m_len,posb->m_inca,posb->m_incb);
//				list.splice(list.end(),list,posa);
//			}
//			else {
//				posa->m_op=new op_loop(posa->m_len,posa->m_inca,posa->m_incb);
//				posb->m_op=new op_daxpy(posb->m_len,posb->m_inca,posb->m_incb);
//				list.splice(posb,list,posa);
//			}
			posb->m_op=new op_daxpby_trp(posa->m_len,posb->m_len,posb->m_inca,posa->m_incb);
			list.erase(posa);
		}
	} catch ( std::bad_alloc& e ) {
		clean_list(list);
		throw out_of_memory("libtensor",k_clazz,
			"build_list(loop_list_t&,const dimensions<N>&,const permutation<N>&,const dimensions<N>&)",
			__FILE__,__LINE__,e.what());
	}
}

template<size_t N>
void tod_add<N>::clean_list( loop_list_t &list )
{
	for (typename loop_list_t::iterator it=list.begin(); it != list.end(); it++) {
		delete it->m_op;
		it->m_op=NULL;
	}
}

template<size_t N>
void tod_add<N>::op_loop::exec( processor_t &proc, registers &regs)
	throw(exception)
{
	const double* ptra=regs.m_ptra;
	double *ptrb = regs.m_ptrb;

	for(size_t i=0; i<m_len; i++) {
		regs.m_ptra=ptra;
		regs.m_ptrb = ptrb;
		proc.process_next();
		ptra+= m_inca;
		ptrb += m_incb;
	}
}


template<size_t N>
const char* tod_add<N>::op_daxpy::k_clazz = "tod_add<N>::op_daxpy";


template<size_t N>
void tod_add<N>::op_daxpy::exec( processor_t &proc, registers &regs)
	throw(exception)
{
	tod_add<N>::op_daxpy::start_timer();
	cblas_daxpy(m_len,regs.m_ca,regs.m_ptra,m_inca,regs.m_ptrb,m_incb);
	tod_add<N>::op_daxpy::stop_timer();
}


template<size_t N>
const char* tod_add<N>::op_daxpby_trp::k_clazz = "tod_add<N>::op_daxpby_trp";


template<size_t N>
void tod_add<N>::op_daxpby_trp::exec( processor_t &proc, registers &regs)
	throw(exception)
{
	tod_add<N>::op_daxpby_trp::start_timer();
	blas::daxpby_trp(regs.m_ptra, regs.m_ptrb, m_leni, m_lenj,
		m_incaj, m_incbi, regs.m_ca, 1.0);
	tod_add<N>::op_daxpby_trp::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_ADD_H

