#ifndef LIBTENSOR_LOOP_LIST_ELEM_H
#define LIBTENSOR_LOOP_LIST_ELEM_H

#include "loop_list_base.h"

namespace libtensor {


/**	\brief Operates nested loops on three arrays with elementwise multiplication
		or division as the kernel (c (+)= a * b or c (+)= a / b)

	\ingroup libtensor_tod
 **/
class loop_list_elem : public loop_list_base<2, 1, loop_list_elem> {
public:
	static const char *k_clazz; //!< Class name

private:
	//!	c_i (+)= k * a_i {*,/} b_i
	struct {
		double m_k;
		size_t m_n;
		size_t m_stepa;
		size_t m_stepb;
	} m_op;

protected:
	void run_loop(list_t &loop, registers &r, double c, bool doadd, bool recip);

private:
	void fn_mult_add(registers &r) const;
	void fn_mult_put(registers &r) const;
	void fn_div_add(registers &r) const;
	void fn_div_put(registers &r) const;
};


} // namespace libtensor

#endif // LIBTENSOR_LOOP_LIST_MULEL_H
