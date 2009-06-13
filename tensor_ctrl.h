#ifndef LIBTENSOR_TENSOR_CTRL_H
#define LIBTENSOR_TENSOR_CTRL_H

#include "defs.h"
#include "exception.h"
#include "tensor_i.h"

namespace libtensor {

/**	\brief Tensor control

	Tensor control keeps track of pointers which have been checked out and
	returns all pointers as soon as it is destructed. Thus, pointers to 
	tensor data are only valid as long as the tensor_ctrl object exist by 
	which they have been requested. 

	\param N Tensor order.
	\param T Tensor element type.

	\ingroup libtensor
**/
template<size_t N, typename T>
class tensor_ctrl {
private:
	struct ptr_node {
		const T* m_ptr;
		size_t m_ptrcnt; 
		ptr_node* m_next;

		ptr_node() : m_ptr(NULL), m_ptrcnt(0), m_next(NULL) {}
	};

	tensor_i<N,T> &m_t; //!< Controlled tensor
	ptr_node m_head; //!< list of data pointers 
	
public:
	//!	\name Construction and destruction
	//@{
	tensor_ctrl(tensor_i<N,T> &t);
	~tensor_ctrl();
	//@}

	//!	\name Event forwarding
	//@{
	void req_prefetch() throw(exception);
	T *req_dataptr() throw(exception);
	const T *req_const_dataptr() throw(exception);
	void ret_dataptr(const T *p) throw(exception);
	//@}
};


template<size_t N, typename T>
inline tensor_ctrl<N,T>::tensor_ctrl(tensor_i<N,T> &t) : m_t(t) {
}

template<size_t N, typename T>
inline tensor_ctrl<N,T>::~tensor_ctrl() {
	if ( m_head.m_ptrcnt != 0 ) {
		while ( m_head.m_ptrcnt != 0 ) {
			m_t.on_ret_dataptr(m_head.m_ptr);
			m_head.m_ptrcnt--;
		}
		
		ptr_node* node=m_head.m_next, *tmp;
		while ( node != NULL ) {
			while ( node->m_ptrcnt != 0 ) {
				m_t.on_ret_dataptr(node->m_ptr);
				node->m_ptrcnt--;
			}
			tmp=node;
			node=node->m_next;
			delete tmp;
		}
	}
}

template<size_t N, typename T>
inline void tensor_ctrl<N,T>::req_prefetch() throw(exception) {
	m_t.on_req_prefetch();
}

template<size_t N, typename T>
inline T *tensor_ctrl<N,T>::req_dataptr() throw(exception) {
	T* ptr=m_t.on_req_dataptr();

	if ( m_head.m_ptrcnt == 0 ) {	
		m_head.m_ptr=ptr;
		m_head.m_ptrcnt++;
	}
	else {
		ptr_node* node=&m_head;
		while ( (node->m_next!=NULL) && ( node->m_ptr!=ptr ) ) node=node->m_next;
		
		if ( node->m_ptr==ptr ) {
			node->m_ptrcnt++;
		}
		else {
			node->m_next=new ptr_node();
			node->m_next->m_ptr=ptr;
			node->m_next->m_ptrcnt++;
		}
	}
	
	return ptr;
}

template<size_t N, typename T>
inline const T *tensor_ctrl<N,T>::req_const_dataptr() throw(exception) {
	const T* ptr=m_t.on_req_const_dataptr();

	if ( m_head.m_ptrcnt == 0 ) {	
		m_head.m_ptr=ptr;
		m_head.m_ptrcnt++;
	}
	else {
		ptr_node* node=&m_head;
		while ( ( node->m_next!=NULL ) && ( node->m_ptr!=ptr ) ) node=node->m_next;
		
		if ( node->m_ptr == ptr ) {
			node->m_ptrcnt++;
		}
		else {
			node->m_next=new ptr_node();
			node->m_next->m_ptr=ptr;
			node->m_next->m_ptrcnt++;
		}
	}
	
	return ptr;
}

template<size_t N, typename T>
inline void tensor_ctrl<N,T>::ret_dataptr(const T *p) throw(exception) {
	if ( m_head.m_ptrcnt == 0 ) 
		throw_exc("tensor_ctrl<N,T>", "ret_dataptr()",
			"No pointer has been checked out."); 

	// traverse the list if the returned pointer is not the main pointer
	if ( m_head.m_ptr != p ) {
		if ( m_head.m_next == NULL ) 
			throw_exc("tensor_ctrl<N,T>", "ret_dataptr()",
				"Invalid data pointer."); 
		
		ptr_node* node=m_head.m_next, *prev=&m_head;
		while (( node->m_next!=NULL ) && ( node->m_ptr!=p )) {
			prev=node;
			node=node->m_next;
		}

		if ( node->m_next==NULL ) 
			throw_exc("tensor_ctrl<N,T>", "ret_dataptr()",
				"Invalid data pointer."); 
		else {
			if ( node->m_ptrcnt == 1 ) {
				prev->m_next=node->m_next;
				delete node;
			}
			else {
				node->m_ptrcnt--;
			}
		}
	}
	else {
		m_head.m_ptrcnt--;
		if ( (m_head.m_ptrcnt==0) && (m_head.m_next!=NULL) ) {
			ptr_node* node  = m_head.m_next;
			m_head.m_ptr    = node->m_ptr;
			m_head.m_ptrcnt = node->m_ptrcnt;
			m_head.m_next=node->m_next;
			delete node;
		}
	}
	m_t.on_ret_dataptr(p);
}

} // namespace libtensor

#endif // LIBTENSOR_TENSOR_CTRL_H

