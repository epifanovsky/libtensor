#ifndef LIBTENSOR_LOOP_LIST_ADD_H
#define LIBTENSOR_LOOP_LIST_ADD_H

#include <list>
#include "loop_list_base.h"

namespace libtensor {


/**	\brief Operates nested loops on two arrays with addition and assignment
		as the kernel (b = a1 + a2)

	\ingroup libtensor_tod
 **/
class loop_list_add : public loop_list_base<1, 1, loop_list_add> {
public:
	static const char *k_clazz; //!< Class name

private:
	struct {
		double m_k;
		size_t m_n;
		size_t m_stepa;
		size_t m_stepb;
	} m_add;

protected:
	void run_loop(list_t &loop, registers &r, double c);

private:
	void install_kernel(list_t &loop, double c);
	void fn_daxpy(registers &r) const;
};


} // namespace libtensor

#endif // LIBTENSOR_LOOP_LIST_ADD_H
