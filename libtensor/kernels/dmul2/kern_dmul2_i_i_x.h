#ifndef LIBTENSOR_KERN_DMUL2_I_I_X_H
#define LIBTENSOR_KERN_DMUL2_I_I_X_H

#include "../kern_dmul2.h"

namespace libtensor {


/** \brief Specialized kernel for \f$ c_i = c_i + a_i b \f$

 	\ingroup libtensor_kernels
 **/
class kern_dmul2_i_i_x : public kernel_base<2, 1> {
	friend class kern_dmul2_i_pi_p;
	friend class kern_dmul2_ij_i_j;
	friend class kern_dmul2_ij_j_i;

public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni;
	size_t m_sia, m_sic;

public:
	virtual ~kern_dmul2_i_i_x() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_dmul2 &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_DMUL2_I_I_X_H
