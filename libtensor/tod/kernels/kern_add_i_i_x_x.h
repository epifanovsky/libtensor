#ifndef LIBTENSOR_KERN_ADD_I_I_X_X_H
#define LIBTENSOR_KERN_ADD_I_I_X_X_H

#include "kern_add_generic.h"

namespace libtensor {


/**	\brief Kernel for \f$ c_i = c_i + (a_i + b) d \f$

 	\ingroup libtensor_tod_kernel
 **/
class kern_add_i_i_x_x : public kernel_base<2, 1> {
public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni;
	size_t m_sia, m_sic;

public:
	virtual ~kern_add_i_i_x_x() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_add_generic &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_ADD_I_I_X_X_H
