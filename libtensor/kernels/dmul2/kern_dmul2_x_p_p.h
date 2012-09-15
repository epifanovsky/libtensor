#ifndef LIBTENSOR_KERN_DMUL2_X_P_P_H
#define LIBTENSOR_KERN_DMUL2_X_P_P_H

#include "../kern_dmul2.h"

namespace libtensor {


template<typename LA> class kern_dmul2_i_ip_p;
template<typename LA> class kern_dmul2_i_p_ip;
template<typename LA> class kern_dmul2_x_pq_qp;


/** \brief Specialized kernel for \f$ c = c + a_p b_p \f$
    \tparam LA Linear algebra.

	\ingroup libtensor_kernels
 **/
template<typename LA>
class kern_dmul2_x_p_p : public kernel_base<LA, 2, 1> {
	friend class kern_dmul2_i_ip_p<LA>;
	friend class kern_dmul2_i_p_ip<LA>;
	friend class kern_dmul2_x_pq_qp<LA>;

public:
	static const char *k_clazz; //!< Kernel name

public:
    typedef typename kernel_base<LA, 2, 1>::device_context_ref
        device_context_ref;
    typedef typename kernel_base<LA, 2, 1>::list_t list_t;
    typedef typename kernel_base<LA, 2, 1>::iterator_t iterator_t;

private:
	double m_d;
	size_t m_np;
	size_t m_spa, m_spb;

public:
	virtual ~kern_dmul2_x_p_p() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

    virtual void run(device_context_ref ctx, const loop_registers<2, 1> &r);

	static kernel_base<LA, 2, 1> *match(const kern_dmul2<LA> &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_X_P_P_H
